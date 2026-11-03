"""Coordinate frame ids and the nominal camera <- device rotation for phone captures.

android_device (D): x right, y toward the top edge, z out of the screen, in the device's natural orientation.
cam (C): OpenCV camera of the upright image, x right, y down, z forward.
world (W): Z up, floor z = 0. gltf (G): Y up, x_G = (x_W, z_W, -y_W).
"""

import numpy as np

ANDROID_DEVICE = "android_device"
CAM = "cam"

DEVICE_ORIENTATIONS = (0, 90, 180, 270)
BACK_CAMERA_SENSOR_ORIENTATION = 90  # usual for phones; not stored in the capture files

R_GW = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])  # gltf <- world: Z up to Y up


def _rot_z(deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def R_cam_imu_upright(device_orientation_deg: int) -> np.ndarray:
    """Rotation cam <- device for the back camera's upright image.

    `device_orientation_deg` is how far the device is turned clockwise (seen from the screen) from its natural
    orientation, as in Android's OrientationEventListener. At 0 the image x/y/z are device x/-y/-z.
    """
    if device_orientation_deg % 90:
        raise ValueError(f"device orientation must be a multiple of 90, got {device_orientation_deg}")
    R = _rot_z(device_orientation_deg) @ np.diag([1.0, -1.0, -1.0])
    return np.round(R) + 0.0  # exact 0/±1, no -0.0


def device_orientation_from_display(rotation_deg: int) -> int:
    """Device orientation from the clockwise rotation that makes stored frames upright (back camera)."""
    return (rotation_deg - BACK_CAMERA_SENSOR_ORIENTATION) % 360


def device_orientation_from_gravity(grav_dev_mean: np.ndarray) -> int:
    """The orientation whose camera frame puts the mean down vector closest to image-down (+y)."""
    return max(DEVICE_ORIENTATIONS, key=lambda o: (R_cam_imu_upright(o) @ grav_dev_mean)[1])
