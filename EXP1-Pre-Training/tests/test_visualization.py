from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

from lakeice_ncde.visualization.pdf_report import build_pdf_report
from lakeice_ncde.visualization.plots import plot_loss_curves


@pytest.mark.skipif(sys.platform.startswith("win"), reason="Matplotlib savefig is unstable under this Windows CI-like environment.")
def test_plot_and_pdf(tmp_path: Path) -> None:
    history = pd.DataFrame(
        {
            "epoch": [1, 2, 3],
            "train_loss": [1.0, 0.8, 0.6],
            "val_loss": [1.2, 0.9, 0.7],
            "val_rmse": [1.0, 0.9, 0.8],
            "val_mae": [0.9, 0.8, 0.7],
            "val_r2": [0.1, 0.2, 0.3],
        }
    )
    figures_dir = tmp_path / "figures"
    plot_loss_curves(history, figures_dir / "01_loss_curve.png")
    assert (figures_dir / "01_loss_curve.png").exists()
    build_pdf_report(figures_dir, figures_dir / "report.pdf")
    assert (figures_dir / "report.pdf").exists()
