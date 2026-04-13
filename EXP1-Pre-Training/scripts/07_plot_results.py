from __future__ import annotations

import argparse
from pathlib import Path

from lakeice_ncde.app import plot_from_run
from lakeice_ncde.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate plots and PDF report for a run.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to a finished run directory.")
    args = parser.parse_args()
    logger = setup_logging()
    plot_from_run(Path(args.run_dir).resolve(), logger)


if __name__ == "__main__":
    main()
