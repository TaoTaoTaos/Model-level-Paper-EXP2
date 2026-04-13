from __future__ import annotations

from pathlib import Path

from lakeice_ncde.app import build_common_parser, resolve_runtime, validate_and_save


def main() -> None:
    parser = build_common_parser("Validate the raw Excel input file.")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config, paths, logger = resolve_runtime(project_root, args.config, args.override, args.set_values)
    validate_and_save(config, paths, logger)


if __name__ == "__main__":
    main()
