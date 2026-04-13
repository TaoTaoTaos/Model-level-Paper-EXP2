from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import optuna
import pandas as pd

from lakeice_ncde.app import resolve_runtime, train_experiment
from lakeice_ncde.config import apply_key_value_overrides, load_config
from lakeice_ncde.utils.io import save_dataframe, save_yaml
from lakeice_ncde.utils.logging import setup_logging
from lakeice_ncde.utils.paths import resolve_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search for NeuralCDE.")
    parser.add_argument("--config", type=str, required=True, help="Path to the Optuna config.")
    parser.add_argument("--override", type=str, action="append", default=[], help="Extra YAML override path.")
    parser.add_argument("--set", dest="set_values", type=str, action="append", default=[], help="Dotted key=value override.")
    parser.add_argument("--retrain-best", action="store_true", help="Retrain a formal run using the best parameters.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    base_config = load_config(
        project_root=project_root,
        config_path=(project_root / args.config).resolve(),
        override_paths=[(project_root / override).resolve() for override in args.override],
    )
    base_config = apply_key_value_overrides(base_config, args.set_values)
    paths = resolve_paths(base_config, project_root)
    logger = setup_logging()

    optuna_cfg = base_config["optuna"]
    study_root = paths.output_root / "optuna_studies" / optuna_cfg["study_name"]
    study_root.mkdir(parents=True, exist_ok=True)
    trial_rows: list[dict] = []

    def objective(trial: optuna.Trial) -> float:
        config = copy.deepcopy(base_config)
        space = optuna_cfg["search_space"]
        config["window"]["window_days"] = trial.suggest_categorical("window_days", space["window_days"])
        config["coeffs"]["interpolation"] = trial.suggest_categorical("interpolation", space["interpolation"])
        config["model"]["hidden_channels"] = trial.suggest_categorical("hidden_channels", space["hidden_channels"])
        config["model"]["hidden_hidden_channels"] = trial.suggest_categorical(
            "hidden_hidden_channels", space["hidden_hidden_channels"]
        )
        config["model"]["num_hidden_layers"] = trial.suggest_categorical(
            "num_hidden_layers", space["num_hidden_layers"]
        )
        config["model"]["dropout"] = trial.suggest_categorical("dropout", space["dropout"])
        config["train"]["learning_rate"] = trial.suggest_float(
            "learning_rate",
            float(space["learning_rate"]["low"]),
            float(space["learning_rate"]["high"]),
            log=bool(space["learning_rate"]["log"]),
        )
        config["train"]["weight_decay"] = trial.suggest_float(
            "weight_decay",
            float(space["weight_decay"]["low"]),
            float(space["weight_decay"]["high"]),
            log=bool(space["weight_decay"]["log"]),
        )
        config["train"]["batch_size"] = trial.suggest_categorical("batch_size", space["batch_size"])
        config["model"]["use_adjoint"] = trial.suggest_categorical("use_adjoint", space["use_adjoint"])
        config["model"]["method"] = trial.suggest_categorical("method", space["method"])
        config["features"]["target_transform"] = trial.suggest_categorical("target_transform", space["target_transform"])
        config["experiment"]["name"] = f"{base_config['experiment']['name']}_trial_{trial.number:03d}"

        trial_output_root = study_root / "runs"
        run_context = train_experiment(config, paths, logger, output_root=trial_output_root)
        summary = json.loads((run_context.run_dir / "run_summary.json").read_text(encoding="utf-8"))
        value = float(summary["best_val_rmse"])
        row = {
            "trial_number": trial.number,
            "value": value,
            "run_dir": str(run_context.run_dir),
            **trial.params,
        }
        trial_rows.append(row)
        save_dataframe(pd.DataFrame(trial_rows), study_root / "optuna_trials.csv")
        return value

    sampler = optuna.samplers.TPESampler(seed=int(optuna_cfg["sampler_seed"]))
    study = optuna.create_study(direction=optuna_cfg["direction"], sampler=sampler, study_name=optuna_cfg["study_name"])
    study.optimize(objective, n_trials=int(optuna_cfg["n_trials"]), timeout=optuna_cfg["timeout"])

    best_params_path = study_root / "best_params.yaml"
    save_yaml(study.best_params, best_params_path)
    logger.info("Best trial value: %.6f", study.best_value)
    logger.info("Best params saved to %s", best_params_path)

    if args.retrain_best or bool(optuna_cfg.get("retrain_best", False)):
        final_config = copy.deepcopy(base_config)
        for key, value in study.best_params.items():
            if key in {"window_days"}:
                final_config["window"]["window_days"] = value
            elif key in {"interpolation"}:
                final_config["coeffs"]["interpolation"] = value
            elif key in {"hidden_channels", "hidden_hidden_channels", "num_hidden_layers", "dropout", "use_adjoint", "method"}:
                final_config["model"][key] = value
            elif key in {"learning_rate", "weight_decay", "batch_size"}:
                final_config["train"][key] = value
            elif key == "target_transform":
                final_config["features"]["target_transform"] = value
        final_config["experiment"]["name"] = f"{base_config['experiment']['name']}_best"
        train_experiment(final_config, paths, logger, output_root=study_root / "best_run")


if __name__ == "__main__":
    main()
