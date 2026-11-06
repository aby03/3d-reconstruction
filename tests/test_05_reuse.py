import dataclasses
from pathlib import Path

import pytest

from recon.dense import DenseOptions
from recon.pipeline.stages import s05_dense
from recon.pipeline.stages.s05_dense import CHAIN_FILE, CHAIN_KEEP, _chain_key, _clear_chain, _load_chain


def _key(options: DenseOptions, tmp: Path) -> str:
    return _chain_key(options, tmp)


def test_chain_key_follows_the_openmvs_inputs_only(tmp_path):
    base = _key(DenseOptions(), tmp_path)
    assert _key(DenseOptions(), tmp_path) == base
    # later steps (depth, fusion) do not change it
    other = DenseOptions(depth_source="simplerecon")
    other.fusion.min_weight = 5.0
    other.mono.edge_rel = 0.1
    other.simplerecon.scale = 3.0
    assert _key(other, tmp_path) == base
    # the OpenMVS steps and the aligned model do
    for change in (
        lambda o: setattr(o.openmvs, "resolution_level", 2),
        lambda o: setattr(o.filter, "enabled", not o.filter.enabled),
        lambda o: setattr(o.mesh, "remove_spurious", 7),
        lambda o: setattr(o.refine, "enabled", not o.refine.enabled),
        lambda o: setattr(o.preview, "enabled", not o.preview.enabled),
    ):
        o = DenseOptions()
        change(o)
        assert _key(o, tmp_path) != base
    (tmp_path / "stage_manifest.json").write_text("{}")
    assert _key(DenseOptions(), tmp_path) != base


def test_load_chain_needs_the_file_the_key_and_every_kept_output(tmp_path):
    assert _load_chain(tmp_path, "k") is None
    (tmp_path / CHAIN_FILE).write_text('{"key": "other"}')
    assert _load_chain(tmp_path, "k") is None
    (tmp_path / CHAIN_FILE).write_text('{"key": "k"}')
    assert _load_chain(tmp_path, "k") is None  # the kept files are missing


def test_clear_chain_removes_the_kept_files_and_nothing_else(tmp_path):
    for rel in CHAIN_KEEP:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir() if "." not in Path(rel).name else path.write_text("x")
    (tmp_path / "reports").mkdir()
    _clear_chain(tmp_path)
    assert [p.name for p in tmp_path.iterdir()] == ["reports"]
