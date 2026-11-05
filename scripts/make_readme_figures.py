"""README figures. The report stage renders them for every run (`08_report/figures/`); this copies a run's figures
into `results/<scene>/` for the README, or builds the tutorial figure:

    pixi run python scripts/make_readme_figures.py --run outputs/hotel1/runs/<run_id>   # -> results/hotel1/*.png
    pixi run python scripts/make_readme_figures.py --tutorials                          # results/tutorial_*.png -> results/tutorials.png

To make the figures of a finished run again: `pixi run recon run --input data/<scene> --resume <run_id> --only report --force report`.
"""

import argparse
import shutil
from pathlib import Path

from recon.report.figures import compose

RESULTS = Path(__file__).resolve().parents[1] / "results"
TUTORIALS = [("tutorial_1b_epipolar", "1b: epipolar geometry"), ("tutorial_1c_matches", "1c: feature matches")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, help="run directory whose 08_report/figures are copied to results/<scene>/")
    parser.add_argument("--tutorials", action="store_true", help="build only the tutorial figure")
    args = parser.parse_args()
    if args.tutorials:
        print(compose(RESULTS / "tutorials.png", 1, 900, [(RESULTS / f"{stem}.png", title) for stem, title in TUTORIALS]))
        return
    if args.run is None:
        parser.error("give --run RUN_DIR or --tutorials")
    scene = args.run.parents[1].name
    out = RESULTS / scene
    out.mkdir(parents=True, exist_ok=True)
    for figure in sorted((args.run / "08_report/figures").glob("*.png")):
        print(shutil.copy(figure, out / figure.name))


if __name__ == "__main__":
    main()
