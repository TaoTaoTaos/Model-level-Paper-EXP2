from __future__ import annotations

import argparse
from pathlib import Path

from lakeice_ncde.app import make_split_artifacts, resolve_runtime, train_experiment
from lakeice_ncde.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leave-one-lake-out cross-validation.")
    parser.add_argument("--config", type=str, required=True, help="LOLO config path.")
    parser.add_argument("--override", type=str, action="append", default=[], help="Extra YAML override path.")
    parser.add_argument("--set", dest="set_values", type=str, action="append", default=[], help="Dotted key=value override.")
    parser.add_argument("--limit-folds", type=int, default=None, help="Optional limit for the number of LOLO folds.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    config, paths, logger = resolve_runtime(project_root, args.config, args.override, args.set_values)
    split_manifests = make_split_artifacts(config, paths, logger)

    selected_manifests = split_manifests[: args.limit_folds] if args.limit_folds is not None else split_manifests
    logger = setup_logging()
    for manifest_path in selected_manifests:
        split_name = manifest_path.parent.name
        logger.info("Running LOLO fold: %s", split_name)
        train_experiment(
            config=config,
            paths=paths,
            logger=logger,
            split_name=split_name,
            output_root=paths.output_root / config["experiment"]["name"],
            split_manifest_path=manifest_path,
        )


if __name__ == "__main__":
    main()
