"""SimpleRecon depth for the keyframes of a run's dense stage, scored against the SfM points the views see.

Search the scale and print the agreement curve (model units per metre; the dense stage does the same with
`dense.depth_source=simplerecon`):

    pixi run -e simplerecon python scripts/simplerecon_depth.py --run RUN

Score given scales on a sample of views, or write the depth of every view (depth in SfM units, `confidence` and
`mono_fit.csv` rows: the layout of the mono depth) for one scale:

    pixi run -e simplerecon python scripts/simplerecon_depth.py --run RUN --scales 0.5 1 2 --views 20
    pixi run -e simplerecon python scripts/simplerecon_depth.py --run RUN --scales 1.7 --write OUT_DIR
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pycolmap
import torch

from recon.dense.mono import MonoOptions
from recon.dense.simplerecon import DepthRunner, SimpleReconOptions, find_scale, start_scale
from recon.io.tables import write_npz
from recon.pipeline.stages import s05_dense


def main() -> None:
    d = SimpleReconOptions()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="run directory with a dense stage")
    parser.add_argument("--scales", type=float, nargs="+", help="scales to score (default: search)")
    parser.add_argument("--accelerometer", type=float, help="start of the search (default: median SfM depth)")
    parser.add_argument("--views", type=int, default=d.search_views, help="views scored per scale (evenly spaced)")
    parser.add_argument("--rotate", action="store_true", help="portrait views turned to landscape")
    parser.add_argument("--long-side", type=int, nargs="+", default=[d.long_side])
    parser.add_argument("--sources", type=int, default=d.num_sources)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--write", type=Path, help="write all views' depth here (needs one scale)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    dense = args.run / s05_dense.DIR_NAME
    rec = pycolmap.Reconstruction(dense / s05_dense.ARTIFACTS["undistorted_model"])
    images_dir = dense / s05_dense.ARTIFACTS["undistorted_images"]
    mono = pd.read_csv(dense / s05_dense.MONO_ARTIFACTS["mono_fit"]).set_index("image")
    rows, written = [], []
    for long_side in args.long_side:
        options = SimpleReconOptions(
            num_sources=args.sources, long_side=long_side, rotate=args.rotate, fp16=args.fp16, search_views=args.views
        )
        runner = DepthRunner(rec, images_dir, options, args.device, MonoOptions())
        if args.scales is None:
            start, why = start_scale(rec, args.accelerometer, options)
            best, curve = find_scale(runner, start, options)
            print(f"start {start:.3g} ({why}) -> scale {best:.3g}")
            print(pd.DataFrame(curve).to_string(index=False))
            scales = [best]
        else:
            scales = args.scales
        picks = np.unique(np.linspace(0, len(runner.names) - 1, min(args.views, len(runner.names))).round().astype(int))
        if args.write:
            picks = np.arange(len(runner.names))
        for scale in scales:
            runner.seconds.clear()
            results = {int(i): runner.depth(int(i), scale) for i in picks}
            fits = [r.fit for r in results.values()]
            err = np.array([f.median_rel_residual_all for f in fits])
            rows.append(
                {
                    "scale": round(scale, 3),
                    "size": "x".join(map(str, next(iter(runner.nets)))),
                    "s/frame": round(float(np.mean(runner.seconds[1:] or runner.seconds)), 3),
                    "VRAM GB": round(runner.peak_vram_gb(), 2),
                    "err all": round(float(np.nanmedian(err)), 4),
                    "inlier": round(float(np.mean([f.inlier_ratio for f in fits])), 3),
                    "fitted": f"{sum(f.fitted for f in fits)}/{len(fits)}",
                    "depth/SfM": round(float(np.nanmedian([r.depth_ratio for r in results.values()])), 3),
                    "conf": round(float(np.mean([(r.confidence > 0).mean() for r in results.values()])), 3),
                    "mono err": round(float(np.nanmedian(mono.loc[[runner.names[i] for i in results], "median_rel_residual"])), 4),
                }
            )
            print(pd.DataFrame(rows[-1:]).to_string(header=len(rows) == 1, index=False), flush=True)
            if args.write:
                for i, r in results.items():
                    name = runner.names[i]
                    write_npz(
                        args.write / s05_dense.MONO_ARTIFACTS["mono"] / Path(name).with_suffix(".npz"),
                        True,
                        depth=r.depth.astype(np.float16),
                        confidence=r.confidence,
                    )
                    written.append({"image": name, **dataclasses.asdict(r.fit)})
    if args.write:
        out = args.write / s05_dense.MONO_ARTIFACTS["mono_fit"]
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(written).to_csv(out, index=False)
        print(f"wrote {len(written)} views to {args.write}")


if __name__ == "__main__":
    main()
