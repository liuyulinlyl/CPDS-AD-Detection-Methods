"""Combine the six UMAP attack-scenario figures into a 2 x 3 panel figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent

PANEL_CONFIG = (
    ("UMAP_windows_visualization_A.png", "(a) Additive FDI attack scenario"),
    ("UMAP_windows_visualization_S.png", "(b) Subtractive FDI attack scenario"),
    ("UMAP_windows_visualization_R.png", "(c) Replay attack scenario"),
    ("UMAP_windows_visualization_RD.png", "(d) Ramp-drift FDI attack scenario"),
    (
        "UMAP_windows_visualization_LMC.png",
        "(e) Local model-consistent FDI attack scenario",
    ),
    (
        "UMAP_windows_visualization_AL.png",
        "(f) Adversarial-learning-based FDI attack scenario",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine six UMAP figures into a two-row, three-column figure."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Directory containing the six input PNG files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "UMAP_windows_visualization_combined.png",
        help="Combined output image path.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output resolution.")
    return parser.parse_args()


def combine_figures(input_dir: Path, output_path: Path, dpi: int) -> None:
    input_dir = input_dir.resolve()
    image_paths = [input_dir / filename for filename, _ in PANEL_CONFIG]
    missing_paths = [path for path in image_paths if not path.is_file()]
    if missing_paths:
        missing_text = "\n".join(f"  - {path}" for path in missing_paths)
        raise FileNotFoundError(f"Missing input figure(s):\n{missing_text}")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
        }
    )

    figure = plt.figure(figsize=(18, 12), facecolor="white")
    grid = figure.add_gridspec(
        nrows=4,
        ncols=3,
        height_ratios=(1.0, 0.075, 1.0, 0.075),
        left=0.01,
        right=0.99,
        bottom=0.015,
        top=0.995,
        wspace=0.015,
        hspace=0.01,
    )

    opened_images: list[Image.Image] = []
    try:
        for index, ((_, title), image_path) in enumerate(zip(PANEL_CONFIG, image_paths)):
            panel_row = (index // 3) * 2
            panel_column = index % 3

            image = Image.open(image_path).convert("RGB")
            opened_images.append(image)

            image_axis = figure.add_subplot(grid[panel_row, panel_column])
            image_axis.imshow(image)
            image_axis.set_axis_off()

            title_axis = figure.add_subplot(grid[panel_row + 1, panel_column])
            title_axis.set_axis_off()
            title_axis.text(
                0.5,
                0.55,
                title,
                ha="center",
                va="center",
                fontsize=20,
            )

        output_path = output_path.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_path,
            dpi=dpi,
            facecolor="white",
            bbox_inches="tight",
            pad_inches=0.03,
        )
        print(f"Combined figure saved to: {output_path}")
    finally:
        plt.close(figure)
        for image in opened_images:
            image.close()


def main() -> None:
    args = parse_args()
    combine_figures(args.input_dir, args.output, args.dpi)


if __name__ == "__main__":
    main()
