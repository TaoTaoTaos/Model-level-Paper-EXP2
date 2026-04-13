from __future__ import annotations

from pathlib import Path

from lakeice_ncde.app import build_common_parser, resolve_runtime, train_experiment


def main() -> None:
    parser = build_common_parser("Run the end-to-end smoke test on a small debug setup.")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    override_paths = list(args.override)
    if "configs/debug/debug_quick.yaml" not in override_paths:
        override_paths.append("configs/debug/debug_quick.yaml")
    config, paths, logger = resolve_runtime(project_root, args.config, override_paths, args.set_values)
    train_experiment(config, paths, logger)


if __name__ == "__main__":
    main()
