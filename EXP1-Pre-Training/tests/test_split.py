from __future__ import annotations

import pandas as pd

from lakeice_ncde.data.split import greedy_group_split, resolve_split_runtime


def test_greedy_group_split_keeps_groups_intact() -> None:
    df = pd.DataFrame(
        {
            "lake_name": ["A"] * 10 + ["B"] * 8 + ["C"] * 6 + ["D"] * 4,
            "value": range(28),
        }
    )
    assignments = greedy_group_split(df, "lake_name", 0.7, 0.15, 0.15, seed=42)
    assert set(assignments.keys()) == {"A", "B", "C", "D"}
    assert set(assignments.values()).issubset({"train", "val", "test"})
    for lake_name in df["lake_name"].unique():
        assert assignments[lake_name] in {"train", "val", "test"}


def test_greedy_group_split_honors_forced_and_allowed_constraints() -> None:
    df = pd.DataFrame(
        {
            "lake_name": ["[Huo]Lake Hnagzhang"] * 4 + ["[Huo]Lake Ulansu"] * 5 + ["[Li]Lake Xiaoxingkai"] * 6 + ["A"] * 8,
            "value": range(23),
        }
    )

    assignments = greedy_group_split(
        df,
        "lake_name",
        0.7,
        0.15,
        0.15,
        seed=42,
        forced_assignments={"[Huo]Lake Hnagzhang": "test"},
        allowed_splits={
            "[Huo]Lake Ulansu": ["train", "val"],
            "[Li]Lake Xiaoxingkai": ["train", "val"],
        },
    )

    assert assignments["[Huo]Lake Hnagzhang"] == "test"
    assert assignments["[Huo]Lake Ulansu"] in {"train", "val"}
    assert assignments["[Li]Lake Xiaoxingkai"] in {"train", "val"}


def test_greedy_group_split_honors_multiple_forced_assignments() -> None:
    df = pd.DataFrame(
        {
            "lake_name": ["[Huo]Lake Hnagzhang"] * 4 + ["[Huo]Lake Ulansu"] * 5 + ["[Li]Lake Xiaoxingkai"] * 6 + ["A"] * 8,
            "value": range(23),
        }
    )

    assignments = greedy_group_split(
        df,
        "lake_name",
        0.7,
        0.15,
        0.15,
        seed=42,
        forced_assignments={
            "[Huo]Lake Hnagzhang": "test",
            "[Huo]Lake Ulansu": "train",
            "[Li]Lake Xiaoxingkai": "val",
        },
    )

    assert assignments["[Huo]Lake Hnagzhang"] == "test"
    assert assignments["[Huo]Lake Ulansu"] == "train"
    assert assignments["[Li]Lake Xiaoxingkai"] == "val"


def test_resolve_split_runtime_generates_random_seed_once_per_run() -> None:
    config = {
        "split": {
            "name": "default_split",
            "seed": None,
        }
    }

    first = resolve_split_runtime(config)
    second = resolve_split_runtime(config)

    assert isinstance(first["seed"], int)
    assert first["seed"] == second["seed"]
    assert first["name"] == second["name"]
    assert first["name"].startswith("default_split_")


def test_resolve_split_runtime_preserves_fixed_seed_and_name() -> None:
    config = {
        "split": {
            "name": "formal_default",
            "seed": 42,
        }
    }

    runtime = resolve_split_runtime(config)

    assert runtime["seed"] == 42
    assert runtime["name"] == "formal_default"
