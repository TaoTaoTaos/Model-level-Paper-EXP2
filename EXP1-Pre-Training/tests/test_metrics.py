from __future__ import annotations

import numpy as np

from lakeice_ncde.evaluation.metrics import compute_regression_metrics


def test_compute_regression_metrics_basic() -> None:
    y_true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y_pred = np.array([1.0, 2.5, 2.0], dtype=np.float32)
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0
    assert "r2" in metrics
    assert "bias" in metrics
