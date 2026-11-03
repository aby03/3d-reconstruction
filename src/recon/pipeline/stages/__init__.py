from recon.pipeline.stages import (  # noqa: F401  (registers the stages)
    s01_ingest,
    s02_keyframes,
    s03_sfm,
    s04_align,
    s05_dense,
    s06_mesh_post,
    s07_export,
    s08_report,
)
