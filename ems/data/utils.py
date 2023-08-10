from pathlib import Path
import numpy as np
import torch
from fairmotion.ops import conversions

def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def get_split_keyids(path: str, split: str):
    filepath = Path(path) / split
    try:
        with filepath.open("r") as file_split:
            return list(map(str.strip, file_split.readlines()))
    except FileNotFoundError:
        raise NameError(f"'{split}' is not recognized as a valid split.")

def mirror(R):
    y = R[:, :, 1]
    z = R[:, :, 2]
    y[:, 0] *= -1
    z[:, 0] *= -1
    x = np.cross(y, z, axis=-1)

    R_mirror = np.hstack([x, y, z])
    R_mirror = R_mirror.reshape(R_mirror.shape[0], 3, 3)
    R_mirror = R_mirror.transpose((0, 2, 1))

    return R_mirror

def swap_left_right(data):
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()

    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    self_chain = [0, 3, 6, 9, 12, 15]

    for i in range(len(right_chain)):
        aa_r = data[:, right_chain[i], :].copy()
        aa_l = data[:, left_chain[i], :].copy()

        R_r = conversions.A2R(aa_r)
        R_l = conversions.A2R(aa_l)

        A_r = conversions.R2A(mirror(R_r))
        A_l = conversions.R2A(mirror(R_l))

        data[:, right_chain[i], :] = A_l
        data[:, left_chain[i], :] = A_r
    
    for i in range(len(self_chain)):
        aa = data[:, self_chain[i], :].copy()
        R = conversions.A2R(aa)
        A = conversions.R2A(mirror(R))
        data[:, self_chain[i], :] = A

    return data

