from __future__ import annotations

import pytest
import torch

from lakeice_ncde.models.neural_cde import NeuralCDERegressor


@pytest.mark.skipif(pytest.importorskip("torchcde", reason="torchcde not installed") is None, reason="torchcde not installed")
def test_neural_cde_forward_scalar() -> None:
    import torchcde  # type: ignore

    x = torch.tensor(
        [
            [
                [0.0, 0.2, 0.1],
                [0.4, 0.4, 0.3],
                [1.0, 0.5, 0.9],
            ]
        ],
        dtype=torch.float32,
    )
    coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(x).squeeze(0)
    model = NeuralCDERegressor(
        input_channels=3,
        hidden_channels=8,
        hidden_hidden_channels=16,
        num_hidden_layers=2,
        dropout=0.0,
        interpolation="hermite",
        method="rk4",
        use_adjoint=False,
        nonnegative_output=True,
    )
    pred = model(coeffs)
    assert pred.shape == (1,)
