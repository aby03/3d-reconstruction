import dataclasses

import cv2
import numpy as np
import pytest
import yaml
from conftest import assert_stage_ok, rerun_copy
from scipy.spatial.transform import Rotation

from recon.io.tables import read_json
from recon.keyframes.metrics import sharpness, to_gray
from recon.keyframes.priors import cam_rotation_from_device
from recon.pipeline.cli import main
from recon.pipeline.config import DEFAULT_CONFIG, SECTIONS, ConfigError, PipelineConfig, load_config


def test_smoke(smoke_run):
    for seq in smoke_run.inputs:
        assert_stage_ok(smoke_run.stage_dir("keyframes", seq))


def test_rerun_adaptive_with_stills(smoke_run, output_root, tmp_path):
    run_dir = rerun_copy(
        smoke_run, output_root, tmp_path, "--to", "keyframes", "--set", "keyframes.method=adaptive", "--set", "keyframes.stills=true"
    )
    statuses = {s["stage"]: s["status"] for s in read_json(run_dir / "run_manifest.json")["stages"]}
    assert statuses["ingest"] == "skipped" and statuses["keyframes"] == "success"
    seq = next(iter(smoke_run.inputs))
    out = tmp_path / smoke_run.scene_dir.name / "sequences" / smoke_run.stage_dir("keyframes", seq).parent.name / "02_keyframes"
    assert any((out / "images" / "cam1").iterdir())


def test_blur_lowers_sharpness():
    gray = to_gray(np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8))
    assert sharpness(cv2.GaussianBlur(gray, (0, 0), 2.0)) < sharpness(gray)


def test_doctor(capsys):
    assert main(["doctor"]) == 0
    assert "cuda_available" in capsys.readouterr().out


def test_cam_rotation_convention():
    # the device turns by theta about its y axis; seen from the camera (y = -device y) that is a turn about -y,
    # and the stored cam_j_from_cam_i is its inverse
    theta = 0.3
    R_Di_Dj = Rotation.from_rotvec([0.0, theta, 0.0]).as_matrix()
    R_Cj_Ci = cam_rotation_from_device(R_Di_Dj, np.diag([1.0, -1.0, -1.0]))
    assert np.allclose(R_Cj_Ci.T, Rotation.from_rotvec([0.0, -theta, 0.0]).as_matrix())


def test_config_errors():
    with pytest.raises(ConfigError, match=r"keyframes\.no_such_key"):
        load_config(overrides=["keyframes.no_such_key=1"])
    with pytest.raises(ConfigError, match=r"keyframes\.reject\.no_such_key"):
        load_config(overrides=["keyframes.reject.no_such_key=1"])
    with pytest.raises(ConfigError, match=r"keyframes\.uniform_fps"):
        load_config(overrides=["keyframes.uniform_fps=fast"])


def _keys(cls, data=None, prefix=""):
    """Dotted names of the option fields of `cls`, or of the keys of `data` laid over it; nested options recurse."""
    types = {f.name: f.type for f in dataclasses.fields(cls)}
    keys = set()
    for name in types if data is None else data:
        nested = dataclasses.is_dataclass(types.get(name))
        sub = None if data is None else data[name]
        keys |= _keys(types[name], sub, f"{prefix}{name}.") if nested else {prefix + name}
    return keys


def test_default_config_lists_every_key():
    data = yaml.safe_load(DEFAULT_CONFIG.read_text())
    assert {name: _keys(cls, data[name]) for name, cls in SECTIONS.items()} == {
        name: _keys(cls) for name, cls in SECTIONS.items()
    }
    assert load_config() == PipelineConfig()
