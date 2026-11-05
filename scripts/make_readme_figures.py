"""Assemble the README figures from single screenshots: each figure is a grid of titled panels scaled to one width,
so the README carries a few small images instead of many large ones.

    pixi run python scripts/make_readme_figures.py            # results/hotel1/panels/*.png -> results/hotel1/*.png
    pixi run python scripts/make_readme_figures.py --tutorials # results/tutorial_*.png -> results/tutorials.png

A missing panel is drawn as a grey placeholder.
"""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

RESULTS = Path(__file__).resolve().parents[1] / "results"
PANELS = RESULTS / "hotel1" / "panels"
GAP, TITLE_H = 8, 30
FIGURES = {  # figure -> (columns, panel width, [(file stem, title)])
    "hotel1/overview": (1, 1400, [("model", "Textured model (model.glb)")]),
    "hotel1/stages": (
        3,
        560,
        [
            ("input_frames", "input: video keyframes + stills"),
            ("sfm_cameras", "sfm: sparse points, cameras"),
            ("dense_cloud", "dense: fused point cloud"),
            ("mesh_geometry", "mesh_post: flat planes, filled gaps"),
            ("textured_mesh", "export: textured model"),
            ("report", "report: checks and timings"),
        ],
    ),
    "hotel1/details": (
        2,
        700,
        [
            ("detail_texture", "texture detail (full-resolution stills)"),
            ("detail_gap_fill", "gap fill: dense mesh vs. filled"),
            ("detail_planes", "wall flatness after plane snapping"),
            ("detail_qa", "dense-to-mesh distance"),
        ],
    ),
    "tutorials": (
        1,
        900,
        [
            ("../../tutorial_1b_epipolar", "1b: epipolar geometry"),
            ("../../tutorial_1c_matches", "1c: feature matches"),
        ],
    ),
}


def _panel(stem: str, title: str, width: int, height: int | None) -> Image.Image:
    path = PANELS / f"{stem}.png"
    if path.is_file():
        image = Image.open(path).convert("RGB")
        return image.resize((width, round(image.height * width / image.width)), Image.LANCZOS)
    image = Image.new("RGB", (width, height or round(width * 9 / 16)), (200, 200, 200))
    draw = ImageDraw.Draw(image)
    draw.text((width // 2 - 40, image.height // 2), "placeholder", fill=(90, 90, 90), font=ImageFont.load_default(20))
    return image


def make(name: str, columns: int, width: int, panels: list[tuple[str, str]]) -> Path:
    images = [_panel(stem, title, width, None) for stem, title in panels]
    rows = [images[i : i + columns] for i in range(0, len(images), columns)]
    heights = [max(im.height for im in row) for row in rows]
    canvas = Image.new(
        "RGB", (columns * width + (columns + 1) * GAP, sum(heights) + len(rows) * (TITLE_H + GAP) + GAP), (255, 255, 255)
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(16)
    y = GAP
    for row, height, titles in zip(rows, heights, [panels[i : i + columns] for i in range(0, len(panels), columns)]):
        for k, (image, (_, title)) in enumerate(zip(row, titles)):
            x = GAP + k * (width + GAP)
            draw.text((x + 4, y + 6), title, fill=(30, 30, 30), font=font)
            canvas.paste(image, (x, y + TITLE_H))
        y += TITLE_H + height + GAP
    out = RESULTS / f"{name}.png"
    canvas.save(out, optimize=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tutorials", action="store_true", help="build only the tutorial figure")
    args = parser.parse_args()
    for name, (columns, width, panels) in FIGURES.items():
        if (name == "tutorials") == args.tutorials:
            print(make(name, columns, width, panels))


if __name__ == "__main__":
    main()
