import numpy as np
from conftest import assert_stage_ok

from recon.core.frames import R_GW


def test_smoke(smoke_run):
    assert_stage_ok(smoke_run.stage_dir("export"))


def test_r_gw():
    assert np.allclose(R_GW @ [1, 2, 3], [1, 3, -2])  # x_G = (x_W, z_W, -y_W)
    assert np.allclose(R_GW @ R_GW.T, np.eye(3)) and np.isclose(np.linalg.det(R_GW), 1.0)
