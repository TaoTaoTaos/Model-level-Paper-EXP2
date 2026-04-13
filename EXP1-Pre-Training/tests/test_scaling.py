from __future__ import annotations

import numpy as np
import pandas as pd

from lakeice_ncde.data.scaling import fit_feature_scaler, inverse_transform_target, transform_target


def test_target_transform_roundtrip() -> None:
    values = np.array([0.0, 0.1, 1.5, 3.0], dtype=np.float32)
    transformed = transform_target(values, "log1p")
    restored = inverse_transform_target(transformed, "log1p")
    assert np.allclose(values, restored)


def test_feature_scaler_uses_train_stats() -> None:
    train_df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    scaler = fit_feature_scaler(train_df, ["x"], target_transform="none", target_column="y")
    assert abs(scaler.mean["x"] - 2.0) < 1.0e-8
    assert scaler.std["x"] > 0
