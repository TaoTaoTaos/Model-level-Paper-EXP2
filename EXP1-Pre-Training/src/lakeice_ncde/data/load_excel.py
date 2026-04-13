from __future__ import annotations

from math import pi
from pathlib import Path

import numpy as np
import pandas as pd

from lakeice_ncde.data.schema import FeatureSchema


def load_raw_excel(raw_excel_path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Load the raw Excel file."""
    return pd.read_excel(raw_excel_path, sheet_name=sheet_name)


def standardize_dataframe(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, FeatureSchema]:
    """Parse datetimes, add cyclical features, sort rows, and return the feature schema."""
    data_cfg = config["data"]
    feature_cfg = config["features"]

    df = df.copy()
    dt_col = data_cfg["datetime_column"]
    doy_col = data_cfg["doy_column"]
    target_col = data_cfg["target_column"]

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df = df.dropna(subset=[data_cfg["lake_column"], dt_col, target_col]).reset_index(drop=True)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    df[doy_col] = pd.to_numeric(df[doy_col], errors="coerce")
    radians = 2.0 * pi * (df[doy_col].fillna(0.0) / 365.25)
    df["doy_sin"] = np.sin(radians)
    df["doy_cos"] = np.cos(radians)

    numeric_columns = feature_cfg["feature_columns"] + [target_col]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.sort_values([data_cfg["lake_column"], dt_col]).reset_index(drop=True)

    schema = FeatureSchema(
        time_channel=feature_cfg["time_channel_name"],
        feature_columns=feature_cfg["feature_columns"],
        input_channels=[feature_cfg["time_channel_name"], *feature_cfg["feature_columns"]],
        target_column=target_col,
        target_transform=feature_cfg["target_transform"],
    )
    return df, schema
