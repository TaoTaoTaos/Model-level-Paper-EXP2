from __future__ import annotations

from pathlib import Path

from lakeice_ncde.app import build_common_parser, build_window_artifacts, resolve_runtime, resolve_split_manifest_path


def main() -> None:
    parser = build_common_parser("Build irregular lookback windows.")
    parser.add_argument("--split-name", type=str, default=None, help="Optional existing split name to reuse.")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config, paths, logger = resolve_runtime(project_root, args.config, args.override, args.set_values)
    split_manifests = None if args.split_name is None else [resolve_split_manifest_path(paths, args.split_name)]
    build_window_artifacts(config, paths, logger, split_manifests=split_manifests)


if __name__ == "__main__":
    main()
