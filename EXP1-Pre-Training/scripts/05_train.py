from __future__ import annotations

from pathlib import Path

from lakeice_ncde.app import build_common_parser, resolve_runtime, train_experiment


def main() -> None:
    parser = build_common_parser("Train a NeuralCDE experiment run.")
    parser.add_argument("--split-name", type=str, default=None, help="Optional explicit split name.")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config, paths, logger = resolve_runtime(project_root, args.config, args.override, args.set_values)
    train_experiment(config, paths, logger, split_name=args.split_name)


if __name__ == "__main__":
    main()
