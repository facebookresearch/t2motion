from pathlib import Path
import numpy as np
import torch
from fairmotion.ops import conversions

def subsample(num_frames, last_framerate, new_framerate):
    step = int(last_framerate / new_framerate)
    assert step >= 1
    frames = np.arange(0, num_frames, step)
    return frames

def downsample_amass(smpl_data, *, downsample, framerate):
    nframes_total = len(smpl_data["poses"])
    last_framerate = smpl_data["mocap_framerate"].item()

    if downsample:
        frames = subsample(nframes_total, last_framerate=last_framerate, new_framerate=framerate)
    else:
        frames = np.arange(nframes_total)

    duration = len(frames)

    # subsample
    smpl_data = {"poses": torch.from_numpy(smpl_data["poses"][frames]).float(),
                 "trans": torch.from_numpy(smpl_data["trans"][frames]).float()}
    return smpl_data, duration

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


def load_amass_keyid(keyid, amass_path, *, correspondances, downsample, framerate):
    rand_rate = False
    # labels = correspondances[keyid]["labels"]
    #identifier = correspondances[keyid]["identifier"]
    smpl_keyid_path = correspondances[keyid]["path"]
    smpl_datapath = Path(amass_path) / smpl_keyid_path
    # print(smpl_datapath)
    ann_trim = correspondances[keyid]["trim"]
    prev_keyid = correspondances[keyid]["prev"]
    next_keyid = correspondances[keyid]["next"]
    if "sync" in correspondances[keyid]:
        sync_type = correspondances[keyid]["sync"]
    if prev_keyid != -1:
        prev_keyid = str(prev_keyid)
    if next_keyid != -1:
        next_keyid = str(next_keyid)
    try:
        smpl_data = np.load(smpl_datapath)
    except FileNotFoundError:
        return None, False, 0, [], -1, -1

    smpl_data = {x: smpl_data[x] for x in smpl_data.files}
    org_duration = len(smpl_data["poses"])

    if "sync" in correspondances[keyid] and  sync_type == 0:
    #swap
        sample_pose = smpl_data["poses"]
        nframes = sample_pose.shape[0]
        axis_angle_poses = sample_pose.reshape(nframes,-1,3)
        swap_angle_poses = swap_left_right(axis_angle_poses)
        swap_angle_poses = swap_angle_poses.reshape(nframes,-1)
        smpl_data["poses"] = swap_angle_poses
        trans = torch.from_numpy(smpl_data["trans"])
        trans[:,0]*=-1
        smpl_data["trans"] = trans.numpy()

    if "sync" in correspondances[keyid] and sync_type == 1:
    #fast
        if rand_rate:
            acc_rate = np.random.uniform(1.5,3.0)
            acc_rate = float("%.2f"% acc_rate)
            smpl_data, ds_duration = downsample_amass(smpl_data, downsample=downsample, framerate=framerate*acc_rate)
        else:
            smpl_data, ds_duration = downsample_amass(smpl_data, downsample=downsample, framerate=framerate*2.0)
    elif "sync" in correspondances[keyid] and sync_type == 2:
    #slow
        if rand_rate:
            acc_rate = np.random.uniform(0.3,0.5)
            acc_rate = float("%.2f"% acc_rate)
            smpl_data, ds_duration = downsample_amass(smpl_data, downsample=downsample, framerate=framerate*acc_rate)
        else:
            smpl_data, ds_duration = downsample_amass(smpl_data, downsample=downsample, framerate=framerate*0.5)
    else:
        smpl_data, ds_duration = downsample_amass(smpl_data, downsample=downsample, framerate=framerate)

    start = int(ann_trim["start"]*ds_duration)
    end = int(ann_trim["end"]*ds_duration)

    return smpl_data, True, smpl_data["poses"].size(0), [start,end], prev_keyid, next_keyid