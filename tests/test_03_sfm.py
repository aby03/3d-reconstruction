import numpy as np
import pytest
from conftest import assert_stage_ok, cli, rerun_copy

from recon.io.tables import read_json
from recon.pipeline.run_dir import read_manifest
from recon.pipeline.stages.s03_sfm import ARTIFACTS


def test_smoke(smoke_run):
    for seq in smoke_run.inputs:
        assert_stage_ok(smoke_run.stage_dir("sfm", seq))


def test_mapper_fallback_and_loop_detection(smoke_run, output_root, tmp_path):
    """An unreachable registration target runs every mapper; sequential matching runs loop detection when the vocab
    tree is available."""
    run_dir = rerun_copy(
        smoke_run,
        output_root,
        tmp_path,
        "--to",
        "sfm",
        "--set",
        "sfm.mapper.min_registered_ratio=1.01",
        "--set",
        "sfm.matching.strategy=sequential",
    )
    out = run_dir / "03_sfm"
    manifest = assert_stage_ok(out)
    report = read_json(out / ARTIFACTS["report"])
    assert len(report["mapper_attempts"]) == 3
    warned = "no vocab tree: loop detection is off" in manifest.warnings
    assert report["loop_detection"]["enabled"] != warned


@pytest.mark.parametrize("feature_type", ["aliked", "disk", "superpoint"])
def test_learned_features(smoke_run, output_root, tmp_path, feature_type):
    if feature_type != "aliked":
        pytest.importorskip("lightglue")
    try:
        run_dir = rerun_copy(smoke_run, output_root, tmp_path, "--to", "sfm", "--set", f"sfm.features.type={feature_type}")
    except AssertionError:
        manifests = list(tmp_path.glob("*/runs/*/03_sfm"))
        error = (read_manifest(manifests[0]).error or "") if manifests else ""
        if feature_type == "aliked" and any(k in error.lower() for k in ("download", "urlopen", "http", "connection", "resolve")):
            pytest.skip(f"model download failed: {error[:200]}")
        raise
    assert_stage_ok(run_dir / "03_sfm")


def test_resume_errors(smoke_capture, tmp_path):
    assert cli(smoke_capture, tmp_path, "run", "--resume", "no_such_run") == 2
    assert cli(smoke_capture, tmp_path, "run", "--resume") == 2


def test_cli_needs_keyframes(smoke_capture, tmp_path, capsys):
    assert cli(smoke_capture, tmp_path, "sfm") == 1
    assert "run `recon keyframes` first" in capsys.readouterr().out


def _fake_model(centres: dict[str, np.ndarray], rotations: dict[str, np.ndarray] | None = None):
    """A stand-in for a pycolmap model: registered images with a camera centre and a cam_from_world rotation."""
    from types import SimpleNamespace

    rotations = rotations or {}

    def image(i: int, name: str):
        R = rotations.get(name, np.eye(3))
        return SimpleNamespace(
            name=name,
            projection_center=lambda c=centres[name]: c,
            cam_from_world=lambda R=R: SimpleNamespace(rotation=SimpleNamespace(matrix=lambda: R)),
        )

    images = {i: image(i, name) for i, name in enumerate(centres)}
    return SimpleNamespace(images=images, reg_image_ids=lambda: list(images))


def test_continuity_flags_displaced_cameras():
    """A smooth path with one displaced video camera, one displaced still and one rotated camera: each is flagged,
    the rest are not."""
    import pandas as pd
    from scipy.spatial.transform import Rotation

    from recon.sfm.continuity import analyze
    from recon.sfm.priors import GyroRotations

    n = 60
    rng = np.random.default_rng(0)
    t = np.arange(n) * 0.2
    path = np.stack([np.cos(t / 3), np.sin(t / 3), 0.1 * t], axis=1) + rng.normal(0, 1e-3, (n, 3))
    names = [f"s/cam0/{i * 6:06d}.jpg" for i in range(n)]
    centres = {name: path[i].copy() for i, name in enumerate(names)}
    centres[names[30]] = centres[names[30]] + [0.0, 0.0, 3.0]  # displaced video camera
    stills = {f"s/cam1/{i * 6:06d}.jpg": path[i] + rng.normal(0, 1e-3, 3) for i in range(5, n, 10)}
    bad_still = "s/cam1/000246.jpg"
    stills[bad_still] = path[41] + [2.0, 0.0, 0.0]
    centres |= stills
    rotations = {names[45]: Rotation.from_euler("z", 25, degrees=True).as_matrix()}
    rows = [(nm, "s", int(nm.split("/")[-1][:-4]), int(t_i * 1e9)) for nm, t_i in zip(names, t)]
    rows += [(nm, "s", int(nm.split("/")[-1][:-4]), int(t[int(nm.split("/")[-1][:-4]) // 6] * 1e9)) for nm in stills]
    images = pd.DataFrame(rows, columns=["image_name", "sequence_id", "idx", "t_ns"])
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n * 6, 1))  # a still gyro: every relative rotation is the identity
    gyro = {"s": GyroRotations(R_cum=quat, R_cam_imu=np.eye(3))}
    report = analyze(_fake_model(centres, rotations), images, gyro)
    flagged = report["flagged"]
    assert names[30] in flagged and "isolated" in flagged[names[30]]
    assert bad_still in flagged and "still_offset" in flagged[bad_still]
    assert names[45] in flagged and "rotation" in flagged[names[45]]
    assert set(flagged) <= {names[29], names[30], names[31], names[44], names[45], names[46], bad_still}
    clean = {k: v for k, v in centres.items() if k not in (names[30], bad_still)}
    assert analyze(_fake_model(clean), images[images["image_name"].isin(clean)], gyro)["num_flagged"] == 0


def test_time_pairs_follow_capture_time_across_cameras():
    import pandas as pd

    from recon.sfm.matching import _query_images, time_order, time_pairs

    # a video camera every 6 frames and a still camera between them, names sorted cam0 < cam1 as in a real run
    rows = [(f"s/cam0/{i * 6:06d}.jpg", "s", i * 100) for i in range(30)]
    rows += [(f"s/cam1/{i * 6 * 5:06d}.jpg", "s", i * 100 * 5 + 50) for i in range(6)]
    images = pd.DataFrame(rows, columns=["image_name", "sequence_id", "t_ns"])
    order = time_order(images)
    assert order[:3] == ["s/cam0/000000.jpg", "s/cam1/000000.jpg", "s/cam0/000006.jpg"]  # the still sits in time
    pairs = set(time_pairs(order, overlap=3, quadratic=False))
    assert ("s/cam1/000000.jpg", "s/cam0/000006.jpg") in pairs  # a still meets the video frames around it
    assert ("s/cam0/000000.jpg", "s/cam1/000000.jpg") in pairs
    assert all(a != b for a, b in pairs) and len(pairs) == len(set(pairs))
    quad = set(time_pairs(order, overlap=3, quadratic=True))
    assert len(quad) > len(pairs) and (order[0], order[8]) in quad
    queries = _query_images(images, period=10)
    assert queries[:3] == ["s/cam0/000000.jpg", "s/cam0/000060.jpg", "s/cam0/000120.jpg"]
    assert sum("cam1" in q for q in queries) == 6  # every still is a query


def _walk(n: int = 40, dt: float = 0.4):
    """A smooth 2D walk in metres sampled at keyframe times, with its acceleration."""
    t = np.arange(n) * dt
    path = lambda q: np.stack([2 * np.sin(q / 2.5), 1.5 * np.cos(q / 3.5), 0.05 * q], axis=1)  # noqa: E731
    return t, path


def test_motion_windows_recover_scale_and_flag_a_displaced_keyframe():
    import pandas as pd

    from recon.sfm.imu_checks import GRAVITY, analyze_motion

    t, path = _walk()
    h = 1e-3
    scale = 2.5
    centres = {f"s/cam0/{i:06d}.jpg": scale * path(t)[i : i + 1][0] for i in range(len(t))}
    bad = "s/cam0/000020.jpg"
    centres[bad] = centres[bad] + np.array([4.0, 3.0, 2.0])
    samples = np.arange(t[0], t[-1], 0.005)
    acc = (path(samples + h) - 2 * path(samples) + path(samples - h)) / h**2
    down = np.array([0.0, 0.0, -1.0])
    f_dev = acc - GRAVITY * down  # the accelerometer reads acceleration minus gravity
    accel = {"s": pd.DataFrame({"t_ns": (samples * 1e9).astype(np.int64), "x": f_dev[:, 0], "y": f_dev[:, 1], "z": f_dev[:, 2]})}
    images = pd.DataFrame({"image_name": list(centres), "sequence_id": "s", "idx": range(len(t)), "t_ns": (t * 1e9).astype(np.int64)})
    report = analyze_motion(_fake_model(centres), images, accel, {"s": np.eye(3)}, down)
    assert report["windows"] > 5
    assert abs(report["scale"]["median"] - scale) / scale < 0.1
    assert bad in report["flagged"]
    assert set(report["flagged"]) <= {bad}


def test_heading_flags_a_keyframe_whose_rotation_the_magnetometer_contradicts():
    import pandas as pd
    from scipy.spatial.transform import Rotation

    from recon.sfm.imu_checks import analyze_heading

    n = 60
    t = np.arange(n) * 0.2
    R = {f"s/cam0/{i:06d}.jpg": Rotation.from_euler("zyx", [0.3 * t[i], 0.2 * np.sin(t[i]), 0.1 * np.cos(t[i])]).as_matrix() for i in range(n)}
    field, offset = np.array([20.0, -5.0, -40.0]), np.array([3.0, -2.0, 1.0])
    readings = np.array([R[name] @ field + offset for name in R])  # the device reads R m + o
    names = list(R)
    wrong = names[30]
    R_model = dict(R)
    R_model[wrong] = Rotation.from_euler("z", 1.0).as_matrix() @ R[wrong]  # a camera rotated 57 degrees about up
    magnetic = {"s": pd.DataFrame({"t_ns": (t * 1e9).astype(np.int64), "x": readings[:, 0], "y": readings[:, 1], "z": readings[:, 2]})}
    images = pd.DataFrame({"image_name": names, "sequence_id": "s", "idx": range(n), "t_ns": (t * 1e9).astype(np.int64)})
    centres = {name: np.zeros(3) for name in names}
    report = analyze_heading(_fake_model(centres, R_model), images, magnetic, {"s": np.eye(3)})
    assert report["checked"] == n
    assert [o["image"] for o in report["outliers"]] == [wrong]
