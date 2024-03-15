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
"""AFL tests."""


from typing import List, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
from numpy.testing import assert_allclose

from flwr.common import Code, FitRes, Status, parameters_to_ndarrays
from flwr.common.parameter import ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy

from .afl import AFL


def test_afl_num_fit_clients_20_available() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = AFL()
    expected = 20

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=20)

    # Assert
    assert expected == actual


def test_afl_num_fit_clients_19_available() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = AFL()
    expected = 19

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=19)

    # Assert
    assert expected == actual


def test_afl_num_fit_clients_10_available() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = AFL()
    expected = 10

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=10)

    # Assert
    assert expected == actual


def test_afl_num_fit_clients_minimum() -> None:
    """Test num_fit_clients function."""
    # Prepare
    strategy = AFL()
    expected = 9

    # Execute
    actual, _ = strategy.num_fit_clients(num_available_clients=9)

    # Assert
    assert expected == actual


def test_afl_num_evaluation_clients_40_available() -> None:
    """Test num_evaluation_clients function."""
    # Prepare
    strategy = AFL(fraction_evaluate=0.05)
    expected = 2

    # Execute
    actual, _ = strategy.num_evaluation_clients(num_available_clients=40)

    # Assert
    assert expected == actual


def test_afl_num_evaluation_clients_39_available() -> None:
    """Test num_evaluation_clients function."""
    # Prepare
    strategy = AFL(fraction_evaluate=0.05)
    expected = 2

    # Execute
    actual, _ = strategy.num_evaluation_clients(num_available_clients=39)

    # Assert
    assert expected == actual


def test_afl_num_evaluation_clients_20_available() -> None:
    """Test num_evaluation_clients function."""
    # Prepare
    strategy = AFL(fraction_evaluate=0.05)
    expected = 2

    # Execute
    actual, _ = strategy.num_evaluation_clients(num_available_clients=20)

    # Assert
    assert expected == actual


def test_afl_num_evaluation_clients_minimum() -> None:
    """Test num_evaluation_clients function."""
    # Prepare
    strategy = AFL(fraction_evaluate=0.05)
    expected = 2

    # Execute
    actual, _ = strategy.num_evaluation_clients(num_available_clients=19)

    # Assert
    assert expected == actual


def test_aggregation_results() -> None:
    """Test aggregate_fit equivalence between AFL and its inplace version."""
    # Prepare
    weights0_0 = np.random.randn(100, 64)
    weights0_1 = np.random.randn(314, 628, 3)
    weights1_0 = np.random.randn(100, 64)
    weights1_1 = np.random.randn(314, 628, 3)

    results: List[Tuple[ClientProxy, FitRes]] = [
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters([weights0_0, weights0_1]),
                num_examples=1,
                metrics={"loss": 0.4},
            ),
        ),
        (
            MagicMock(),
            FitRes(
                status=Status(code=Code.OK, message="Success"),
                parameters=ndarrays_to_parameters([weights1_0, weights1_1]),
                num_examples=5,
                metrics={"loss": 132},
            ),
        ),
    ]
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]] = []

    afl = AFL()

    # Execute
    result, _ = afl.aggregate_fit(1, results, failures)
    assert result