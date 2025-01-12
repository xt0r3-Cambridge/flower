from typing import List, Tuple, Dict
import random
import time

import flwr as fl
from flwr.server import Driver
from flwr.common import Context

from flwr.common import (
    ServerMessage,
    FitIns,
    ndarrays_to_parameters,
    serde,
    parameters_to_ndarrays,
    ClientMessage,
    NDArrays,
    Code,
)
from flwr.proto import driver_pb2, task_pb2, node_pb2, transport_pb2
from flwr.server.strategy.aggregate import aggregate
from flwr.common import Metrics
from flwr.server import History
from flwr.common import serde
from task import Net, get_parameters, set_parameters
from flwr.common.recordset_compat import fitins_to_recordset, recordset_to_fitres
from flwr.common import Message


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]

    # Multiply accuracy of each client by number of examples used
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_accuracy"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "train_loss": sum(train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }


# Run via `flower-server-app server:app`
app = fl.server.ServerApp()


@app.main()
def main(driver: Driver, context: Context) -> None:
    """."""
    print("RUNNING!!!!!")

    anonymous_client_nodes = False
    num_client_nodes_per_round = 2
    sleep_time = 1
    num_rounds = 3
    parameters = ndarrays_to_parameters(get_parameters(net=Net()))

    history = History()
    for server_round in range(num_rounds):
        print(f"Commencing server round {server_round + 1}")

        # List of sampled node IDs in this round
        sampled_nodes: List[int] = []

        # The Driver API might not immediately return enough client node IDs, so we
        # loop and wait until enough client nodes are available.
        while True:
            all_node_ids = driver.get_node_ids()

            print(f"Got {len(all_node_ids)} client nodes: {all_node_ids}")
            if len(all_node_ids) >= num_client_nodes_per_round:
                # Sample client nodes
                sampled_nodes = random.sample(all_node_ids, num_client_nodes_per_round)
                break
            time.sleep(3)

        # Log sampled node IDs
        print(f"Sampled {len(sampled_nodes)} node IDs: {sampled_nodes}")

        # Schedule a task for all sampled nodes
        fit_ins: FitIns = FitIns(parameters=parameters, config={})
        recordset = fitins_to_recordset(fitins=fit_ins, keep_input=True)

        messages = []
        for node_id in sampled_nodes:
            message = driver.create_message(
                content=recordset,
                message_type="fit",
                dst_node_id=node_id,
                group_id=str(server_round),
                ttl="",
            )
            messages.append(message)

        message_ids = driver.push_messages(messages)
        print(f"Pushed {len(message_ids)} messages: {message_ids}")

        # Wait for results, ignore empty message_ids
        message_ids = [message_id for message_id in message_ids if message_id != ""]

        all_replies: List[Message] = []
        while True:
            replies = driver.pull_messages(message_ids=message_ids)
            print(f"Got {len(replies)} results")
            all_replies += replies
            if len(all_replies) == len(message_ids):
                break
            time.sleep(3)

        # Collect correct results
        all_fitres = [
            recordset_to_fitres(msg.content, keep_input=True) for msg in all_replies
        ]
        print(f"Received {len(all_fitres)} results")

        weights_results: List[Tuple[NDArrays, int]] = []
        metrics_results: List[Tuple[int, Dict]] = []
        for fitres in all_fitres:
            print(f"num_examples: {fitres.num_examples}, status: {fitres.status.code}")

            # Aggregate only if the status is OK
            if fitres.status.code != Code.OK:
                continue
            weights_results.append(
                (parameters_to_ndarrays(fitres.parameters), fitres.num_examples)
            )
            metrics_results.append((fitres.num_examples, fitres.metrics))

        # Aggregate parameters (FedAvg)
        parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
        parameters = parameters_aggregated

        # Aggregate metrics
        metrics_aggregated = weighted_average(metrics_results)
        history.add_metrics_distributed_fit(
            server_round=server_round, metrics=metrics_aggregated
        )
        print("Round ", server_round, " metrics: ", metrics_aggregated)

        # Slow down the start of the next round
        time.sleep(sleep_time)

    print("app_fit: losses_distributed %s", str(history.losses_distributed))
    print("app_fit: metrics_distributed_fit %s", str(history.metrics_distributed_fit))
    print("app_fit: metrics_distributed %s", str(history.metrics_distributed))
    print("app_fit: losses_centralized %s", str(history.losses_centralized))
    print("app_fit: metrics_centralized %s", str(history.metrics_centralized))
