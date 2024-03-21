# Copyright 2024 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Agnostic Federated Learning (AFL) [Mohri et al., 2019] strategy.

Paper: arxiv.org/abs/1902.00146v1
"""

from collections import defaultdict
import numpy as np

from logging import WARNING
from collections.abc import Callable

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Metrics,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .strategy import Strategy
from .utils import project_on_simplex
from .aggregate import aggregate, weighted_loss_avg

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""


# pylint: disable=line-too-long
class AFL(Strategy):
    """Agnostic Federated Learning Strategy.

    Implementation based on https://arxiv.org/abs/1902.00146v1
    This implementation uses projected gradient descent to
    solve the minimax optimisation problem set out in the paper.

    Parameters
    ----------
    lambda_learning_rate: float, optional
        The learning rate for the lambda vector that maximises client loss.
    fraction_fit : float, optional
        Fraction of clients used during training. In case `min_fit_clients`
        is larger than `fraction_fit * available_clients`, `min_fit_clients`
        will still be sampled. Defaults to 1.0.
    fraction_evaluate : float, optional
        Fraction of clients used during validation. In case `min_evaluate_clients`
        is larger than `fraction_evaluate * available_clients`,
        `min_evaluate_clients` will still be sampled. Defaults to 1.0.
    min_fit_clients : int, optional
        Minimum number of clients used during training. Defaults to 2.
    min_evaluate_clients : int, optional
        Minimum number of clients used during validation. Defaults to 2.
    min_available_clients : int, optional
        Minimum number of total clients in the system. Defaults to 2.
    evaluate_fn : Optional[Callable[[int, NDArrays, Dict[str, Scalar]],
        Optional[Tuple[float, Dict[str, Scalar]]]]]
        Optional function used for validation. Defaults to None.
    on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure training. Defaults to None.
    on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
        Function used to configure validation. Defaults to None.
    accept_failures : bool, optional
        Whether or not accept rounds containing failures. Defaults to True.
    initial_parameters : Parameters, optional
        Initial global model parameters.
    fit_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    evaluate_metrics_aggregation_fn : Optional[MetricsAggregationFn]
        Metrics aggregation function, optional.
    """

    # pylint: disable=too-many-arguments,too-many-instance-attributes, line-too-long
    def __init__(
        self,
        *,
        lambda_learning_rate: float = 2e-2,
        return_lambdas: bool = False,
        return_per_client_loss: bool = False,
        lr_schedule: bool = True,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: (
            Callable[
                [int | float, NDArrays, dict[str, Scalar]],
                tuple[float, dict[str, Scalar]] | None,
            ]
            | None
        ) = None,
        on_fit_config_fn: Callable[[int | float], dict[str, Scalar]] | None = None,
        on_evaluate_config_fn: Callable[[int | float], dict[str, Scalar]] | None = None,
        accept_failures: bool = True,
        initial_parameters: Parameters | None = None,
        fit_metrics_aggregation_fn: (
            Callable[[list[tuple[float, Metrics]]], Metrics] | None
        ) = None,
        evaluate_metrics_aggregation_fn: (
            Callable[[list[tuple[float, Metrics]]], Metrics] | None
        ) = None,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.lr_schedule = lr_schedule
        self.return_per_client_loss = return_per_client_loss
        self.lambda_learning_rate = (
            (lambda server_round: lambda_learning_rate / np.sqrt(server_round))
            if lr_schedule
            else (lambda _: lambda_learning_rate)
        )

        # If the model is never initalised, use a defaultdict of 1 for
        # the lambdas.
        # The projection steps will still make sure that the lambdas are
        # normalised
        self.lambdas: defaultdict[str, float] = defaultdict(lambda: 1.0)
        self.return_lambdas = return_lambdas

    def __repr__(self) -> str:
        """Compute a string representation of the strategy."""
        rep = f"AFL(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Return the sample size and the required number of available clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        # Initialize the lambdas to 1 / num_clients
        self.lambdas = defaultdict(
            lambda: 1.0,
            {
                cli.cid: 1.0 / client_manager.num_available()
                for cli in client_manager.all().values()
            },
        )
        return initial_parameters

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> tuple[float, dict[str, Scalar]] | None:
        """Evaluate model parameters using an evaluation function."""
        if self.evaluate_fn is None:
            # No evaluation function provided
            return None
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        eval_res = self.evaluate_fn(server_round, parameters_ndarrays, {})
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:  # noqa: PLR2004
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def project(self) -> None:
        """Project on the simplex lambda_1 + ... + lambda_n = 1."""
        self.lambdas = defaultdict(
            lambda: 1.0,
            zip(
                self.lambdas.keys(),
                project_on_simplex(list(self.lambdas.values())),
                strict=False,
            ),
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        for cli, res in results:
            assert (
                "train_loss" in res.metrics
            ), f'''Loss not found in the results of client {cli}.
AFL requires the models to return the training loss under the key "train_loss"'''

        for cli, res in results:
            self.lambdas[cli.cid] += res.metrics[
                "train_loss"
            ] * self.lambda_learning_rate(server_round)

        self.project()

        # TODO: change this
        # Aggregate the results weighting by the lambdas
        lambda_weighted_results = [
            (parameters_to_ndarrays(fit_res.parameters), self.lambdas[cli.cid])
            for cli, fit_res in results
        ]
        aggregated_ndarrays = aggregate(lambda_weighted_results)

        # Update lambdas
        # As the lambdas change the results linearly in terms of the loss,
        # the gradient vector is just the loss metrics for each of the clients.
        # We treat unselected clients as those with 0 loss

        parameters_aggregated = ndarrays_to_parameters(aggregated_ndarrays)

        # print(self.lambdas)

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated: dict[str, bool | bytes | float | int | str] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(self.lambdas[cli.cid], res.metrics) for cli, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        if self.return_lambdas:
            for k, v in self.lambdas.items():
                metrics_aggregated[f"lambdas_{k}"] = v
        if self.return_per_client_loss:
            for cli, res in results:
                metrics_aggregated[f"client_train_loss_{cli.cid}"] = res.metrics[
                    "train_loss"
                ]
        metrics_aggregated["max_train_loss"] = max(
            [res.metrics["train_loss"] for cli, res in results]
        )

        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[tuple[ClientProxy, EvaluateRes] | BaseException],
    ) -> tuple[float | None, dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Make sure we add everything to the lambdas dict
        _ = [self.lambdas[cli.cid] for cli, _ in results]
        self.project()

        # Aggregate loss
        loss_aggregated = weighted_loss_avg([
            (self.lambdas[cli.cid], evaluate_res.loss) for cli, evaluate_res in results
        ])

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [
                (self.lambdas[cli.cid], res.metrics) for cli, res in results
            ]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated
