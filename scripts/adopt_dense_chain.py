"""Make a finished dense stage reusable: write the `chain.json` that a run made before step reuse lacks, so that
`recon run --resume RUN --from dense ...` with the same OpenMVS options and aligned model skips undistort, OpenMVS
(densify, mesh, refine, preview) and the point filter. Takes the arguments of `recon run` that choose the run and
the config (the key comes from the same config):

    cp -al outputs/hotel1/runs/OLD outputs/hotel1/runs/NEW
    pixi run python scripts/adopt_dense_chain.py --input data/hotel1 --resume NEW
    pixi run -e simplerecon recon run --input data/hotel1 --resume NEW --from dense --set dense.depth_source=simplerecon
"""

import argparse

from recon.pipeline import cli
from recon.pipeline.runner import open_run
from recon.pipeline.stages import s04_align, s05_dense


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli._add_capture_args(parser)
    cli._add_output_args(parser)
    cli._add_resume_arg(parser, "required")
    args = parser.parse_args()
    if not args.resume:
        parser.error("--resume RUN_ID is required")
    hardware = cli._hardware(args)
    ctx = open_run(args.input, cli._config(args, hardware), args.scene, args.resume, hardware)
    out = ctx.stage_dir(s05_dense.STAGE)
    key = s05_dense._chain_key(ctx.config.dense, ctx.stage_dir(s04_align.STAGE))
    s05_dense.adopt_chain(out, key)
    chain = s05_dense._load_chain(out, key)
    print(f"{out}: {'ready to reuse' if chain is not None else 'chain.json written but files are missing'}")


if __name__ == "__main__":
    main()
