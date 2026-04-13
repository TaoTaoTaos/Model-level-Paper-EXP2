from __future__ import annotations

from pathlib import Path

from lakeice_ncde.app import build_common_parser, prepare_dataframe_artifact, resolve_runtime


def main() -> None:
    parser = build_common_parser("Prepare the standardized dataframe artifact.")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config, paths, logger = resolve_runtime(project_root, args.config, args.override, args.set_values)
    prepare_dataframe_artifact(config, paths, logger)


if __name__ == "__main__":
    main()
