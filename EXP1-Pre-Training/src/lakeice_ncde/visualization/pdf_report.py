from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages


matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_pdf_report(figures_dir: Path, pdf_path: Path) -> None:
    """Collect all PNG figures into one PDF report."""
    png_files = sorted(figures_dir.glob("*.png"))
    if not png_files:
        return

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        for png_path in png_files:
            image = mpimg.imread(png_path)
            fig = plt.figure(figsize=(11.69, 8.27))
            ax = fig.add_subplot(111)
            ax.imshow(image)
            ax.set_title(png_path.name)
            ax.axis("off")
            pdf.savefig(fig)
            plt.close(fig)
