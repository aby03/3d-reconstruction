"""Feature type switch of the sfm stage: options, pair lists, keypoint conventions and the database writer. No GPU and
no model download."""

import importlib.util
import sys
import types

import numpy as np
import pandas as pd
import pycolmap
import pytest
from PIL import Image

from recon.pipeline.config import load_config
from recon.sfm import SfmOptions
from recon.sfm.features import FeatureOptions, colmap_max_features, colmap_types, extract_features
from recon.sfm.learned import (
    LearnedFeaturesError,
    check_installed,
    colmap_keypoints,
    extract_learned,
    match_learned,
    resized_size,
)
from recon.sfm.matching import MatchingOptions, candidate_pairs, time_pairs


def test_default_is_sift():
    assert SfmOptions().features.type == "sift"
    assert load_config().sfm.features.type == "sift"
    assert load_config().sfm.features.matcher == "bruteforce"


def test_colmap_type_table():
    E, M = pycolmap.FeatureExtractorType, pycolmap.FeatureMatcherType
    assert colmap_types(FeatureOptions()) == (E.SIFT, M.SIFT_BRUTEFORCE)
    assert colmap_types(FeatureOptions(matcher="lightglue")) == (E.SIFT, M.SIFT_LIGHTGLUE)
    assert colmap_types(FeatureOptions(type="aliked")) == (E.ALIKED_N16ROT, M.ALIKED_BRUTEFORCE)
    assert colmap_types(FeatureOptions(type="aliked", aliked_model="n32", matcher="lightglue")) == (
        E.ALIKED_N32,
        M.ALIKED_LIGHTGLUE,
    )


def test_lightglue_feature_cap():
    assert colmap_max_features(FeatureOptions()) == 8192
    assert colmap_max_features(FeatureOptions(matcher="lightglue")) == 4096
    assert colmap_max_features(FeatureOptions(matcher="lightglue", max_num_features=1024)) == 1024


def test_learned_types_need_lightglue():
    with pytest.raises(ValueError, match="matcher lightglue"):
        FeatureOptions(type="disk")
    with pytest.raises(ValueError, match="matcher lightglue"):
        FeatureOptions(type="superpoint", matcher="bruteforce")
    FeatureOptions(type="disk", matcher="lightglue")


def test_type_validation():
    with pytest.raises(ValueError, match="sfm.features.type"):
        FeatureOptions(type="orb")
    with pytest.raises(ValueError, match="aliked_model"):
        FeatureOptions(type="aliked", aliked_model="n8")
    with pytest.raises(ValueError, match="recon.sfm.learned"):
        extract_features(None, None, [], FeatureOptions(type="disk", matcher="lightglue"))


def test_config_override():
    config = load_config(overrides=["sfm.features.type=superpoint", "sfm.features.matcher=lightglue"])
    assert config.sfm.features.type == "superpoint"


@pytest.mark.skipif(importlib.util.find_spec("lightglue") is not None, reason="lightglue is installed")
def test_missing_package_error():
    with pytest.raises(LearnedFeaturesError, match="pixi run -e lightglue"):
        check_installed()


def test_resized_size():
    assert resized_size(3840, 2160, 1920) == (1920, 1080)
    assert resized_size(1080, 1920, 960) == (540, 960)
    assert resized_size(640, 480, 1920) == (640, 480)


def test_keypoint_scaling_and_centre_offset():
    # no scaling: integer pixel centres become COLMAP's half-integer ones
    np.testing.assert_allclose(colmap_keypoints(np.array([[0.0, 0.0], [9.0, 4.0]]), (10, 5), (10, 5)), [[0.5, 0.5], [9.5, 4.5]])
    # half size: resized pixel 0 covers full pixels 0 and 1, its centre is the full image coordinate 1.0
    out = colmap_keypoints(np.array([[0.0, 0.0], [4.5, 2.0]]), (10, 6), (5, 3))
    np.testing.assert_allclose(out, [[1.0, 1.0], [10.0, 5.0]])
    assert out.dtype == np.float32


def test_candidate_pairs():
    names = [f"s/c/{i:02d}.jpg" for i in range(6)]
    options = MatchingOptions(overlap=2, quadratic_overlap=False)
    assert len(candidate_pairs(names, "exhaustive", options)) == 15
    named = candidate_pairs(names[::-1], "time", options)
    assert named == time_pairs(names, 2, False)
    assert (names[0], names[1]) in named and (names[0], names[3]) not in named
    images = pd.DataFrame({"image_name": names, "sequence_id": "a", "t_ns": [5, 4, 3, 2, 1, 0]})
    timed = candidate_pairs(names, "time", options, images)
    assert (names[5], names[4]) in timed and (names[4], names[5]) not in timed  # earlier capture first
    assert (names[5], names[3]) in timed and (names[5], names[2]) not in timed
    # without a capture time table `time` runs as name-order neighbours
    assert candidate_pairs(names, "time", options) == named


class _FakeExtractor:
    """Stands in for DISK / SuperPoint: a fixed grid of keypoints on the resized image."""

    def extract(self, image, resize=None):
        import torch

        _, h, w = image.shape
        xy = torch.tensor([[1.0, 2.0], [w - 1.0, h - 1.0], [3.0, 3.0]])
        return {"keypoints": xy[None], "descriptors": torch.eye(3, 8)[None]}


def test_database_writing(tmp_path, monkeypatch):
    import torch

    import recon.sfm.learned as learned

    root = tmp_path / "images"
    (root / "seq" / "cam").mkdir(parents=True)
    names = [f"seq/cam/{i}.jpg" for i in range(3)]
    for name in names:
        Image.fromarray(np.random.default_rng(0).integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(root / name)
    monkeypatch.setattr(learned, "check_installed", lambda: None)
    monkeypatch.setattr(learned, "_extractor", lambda options, device: _FakeExtractor())
    database = tmp_path / "db.db"
    options = FeatureOptions(type="superpoint", matcher="lightglue", max_image_size=40)
    extract_learned(database, root, names, options, tmp_path / "feats", use_gpu=False)
    db = pycolmap.Database.open(database)
    try:
        assert db.num_images() == 3 and db.num_cameras() == 1
        for name in names:
            kp = db.read_keypoints(db.read_image_with_name(name).image_id)
            # resized to 40x30: scale 2, so the corner keypoint (39, 29) lands at (79, 59)
            np.testing.assert_allclose(kp[:, :2], [[3.0, 5.0], [79.0, 59.0], [7.0, 7.0]])
        assert db.num_descriptors() == 0
    finally:
        db.close()

    class _FakeMatcher(torch.nn.Module):
        def __init__(self, features):
            super().__init__()
            assert features == "superpoint"

        def forward(self, data):
            assert data["image0"]["keypoints"].shape == (1, 3, 2)
            return {"matches0": torch.tensor([[0, 1, -1]])}

    monkeypatch.setitem(sys.modules, "lightglue", types.SimpleNamespace(LightGlue=_FakeMatcher))
    matching = MatchingOptions(min_num_inliers=2)
    pairs = [(names[2], names[0]), (names[0], names[1])]
    match_learned(database, pairs, options, matching, tmp_path / "feats", tmp_path / "pairs.txt", use_gpu=False)
    written = [tuple(line.split()) for line in (tmp_path / "pairs.txt").read_text().splitlines()]
    assert {frozenset(p) for p in written} == {frozenset(p) for p in pairs} and len(written) == 2
    db = pycolmap.Database.open(database)
    try:
        ids = [db.read_image_with_name(n).image_id for n in names]
        np.testing.assert_array_equal(db.read_matches(ids[0], ids[1]), [[0, 0], [1, 1]])
        assert db.exists_matches(ids[0], ids[2]) and not db.exists_matches(ids[1], ids[2])
    finally:
        db.close()
