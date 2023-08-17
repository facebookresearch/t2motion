import logging

import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from ems.model.metrics import ComputeMetrics
import ems.launch.prepare  # noqa
import numpy as np
import json

logger = logging.getLogger(__name__)

def calc_jitter(joint_p,f=12.5):
    jkp = ((joint_p[3:] - 3 * joint_p[2:-1] + 3 * joint_p[1:-2] - joint_p[:-3]) * (f ** 3)).norm(dim=2) 
    return jkp.mean()/1000

def calc_footskating(joint_p,fps=12.5):
    num_frames = joint_p.shape[0]
    left_foot_joints = joint_p[:,LF]
    right_foot_joints = joint_p[:,RF]
    left_foot_h = left_foot_joints[:,2]
    right_foot_h = right_foot_joints[:,2]
    left_foot_velocity = np.array([np.linalg.norm(left_foot_joints[i,:2]-left_foot_joints[i-1,:2])*fps for i in range(1,num_frames)])
    right_foot_velocity = np.array([np.linalg.norm(right_foot_joints[i,:2]-right_foot_joints[i-1,:2])*fps for i in range(1,num_frames)])
    left_foot_val = np.clip(2 - np.exp2(left_foot_h/0.025),0,1)[1:]
    right_foot_val = np.clip(2 - np.exp2(right_foot_h/0.025),0,1)[1:]
    left_foot_skate = left_foot_val*left_foot_velocity
    right_foot_skate = right_foot_val*right_foot_velocity
    return np.mean((left_foot_skate+right_foot_skate)/2)

def calc_ground_penetrating(joint_p):
    num_frames = joint_p.shape[0]
    left_foot_joints = joint_p[:,LF]
    right_foot_joints = joint_p[:,RF]
    ground_z = min(left_foot_joints[0,-1],right_foot_joints[0,-1])
    left_foot_ground_penetrate = left_foot_joints[:,-1]-ground_z
    right_foot_ground_penetrate = right_foot_joints[:,-1]-ground_z
    sum_val = 0
    for val in left_foot_ground_penetrate:
        if val < 0:
            sum_val+=val
    for val in right_foot_ground_penetrate:
        if val < 0:
            sum_val+=val
    return -sum_val / (len(left_foot_ground_penetrate)+len(right_foot_ground_penetrate))
    

kinematic_tree = [[0, 1, 2, 3, 4], [3, 5, 6, 7], [3, 8, 9, 10],
                  [0, 11, 12, 13, 14, 15],
                  [0, 16, 17, 18, 19, 20]]

mmm_joints = ["root", "BP", "BT", "BLN", "BUN", "LS", "LE", "LW", "RS", "RE", "RW", "LH",
              "LK", "LA", "LMrot", "LF", "RH", "RK", "RA", "RMrot", "RF"]

# Get the indexes of particular body part
# Feet
LM, RM = mmm_joints.index("LMrot"), mmm_joints.index("RMrot")
LF, RF = mmm_joints.index("LF"), mmm_joints.index("RF")

# Shoulders
LS, RS = mmm_joints.index("LS"), mmm_joints.index("RS")
# Hips
LH, RH = mmm_joints.index("LH"), mmm_joints.index("RH")

def matrix_of_angles(cos, sin, inv=False):
    sin = -sin if inv else sin
    return np.stack((np.stack((cos, -sin), axis=-1),
                     np.stack((sin, cos), axis=-1)), axis=-2)

def get_forward_direction(poses):
    across = poses[..., RH, :] - poses[..., LH, :] + poses[..., RS, :] - poses[..., LS, :]
    forward = np.stack((-across[..., 2], across[..., 0]), axis=-1)
    forward = forward/np.linalg.norm(forward, axis=-1)
    return forward

def softmax(x, softness=1.0, dim=None):
    maxi, mini = x.max(dim), x.min(dim)
    return maxi + np.log(softness + np.exp(mini - maxi))


def softmin(x, softness=1.0, dim=0):
    return -softmax(-x, softness=softness, dim=dim)

def get_floor(poses, jointstype="mmm"):
    assert jointstype == "mmm"
    ndim = len(poses.shape)

    foot_heights = poses[..., (LM, LF, RM, RF), 1].min(-1)
    floor_height = softmin(foot_heights, softness=0.5, dim=-1)
    return floor_height[tuple((ndim - 2) * [None])].T

def canonicalize_joints(joints):
    poses = joints.copy()
    translation = joints[..., 0, :].copy()

    # Let the root have the Y translation
    translation[..., 1] = 0
    # Trajectory => Translation without gravity axis (Y)
    trajectory = translation[..., [0, 2]]

    # Remove the floor
    poses[..., 1] -= get_floor(poses)

    # Remove the trajectory of the joints
    poses[..., [0, 2]] -= trajectory[..., None, :]

    # Let the first pose be in the center
    trajectory = trajectory - trajectory[..., 0, :]

    # Compute the forward direction of the first frame
    forward = get_forward_direction(poses[..., 0, :, :])

    # Construct the inverse rotation matrix
    sin, cos = forward[..., 0], forward[..., 1]
    rotations_inv = matrix_of_angles(cos, sin, inv=True)

    # Rotate the trajectory
    trajectory_rotated = np.einsum("...j,...jk->...k", trajectory, rotations_inv)

    # Rotate the poses
    poses_rotated = np.einsum("...lj,...jk->...lk", poses[..., [0, 2]], rotations_inv)
    poses_rotated = np.stack((poses_rotated[..., 0], poses[..., 1], poses_rotated[..., 1]), axis=-1)

    # Re-merge the pose and translation
    poses_rotated[..., (0, 2)] += trajectory_rotated[..., None, :]
    return poses_rotated

def prepare_joints(joints, canonicalize=True, always_on_floor=False):
    # All face the same direction for the first frame
    if canonicalize:
        data = canonicalize_joints(joints)
    else:
        data = joints

    # Rescaling, shift axis and swap left/right
    data = data * 0.75 / 480

    # Swap axis (gravity=Z instead of Y)
    data = data[..., [2, 0, 1]]

    # Make left/right correct
    data[..., [1]] = -data[..., [1]]

    # Center the first root to the first frame
    data -= data[[0], [0], :]

    # Remove the floor
    data[..., 2] -= data[..., 2].min()

    # Put all the body on the floor
    if always_on_floor:
        data[..., 2] -= data[..., 2].min(1)[:, None]

    return data
 
# def foot_penetrating(joint_p):
#     ground = min(joint_p[0,10,-1].item(),joint_p[0,11,-1])
#     left_foot_joints = joint_p[:,10,-1]
#     right_foot_joints = joint_p[:,11,-1]
#     return torch.sum(left_foot_joints-ground)
    

def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)

def sanitize(dico):
    dico = {key: "{:.5f}".format(float(val)) for key, val in dico.items()}
    return dico

def regroup_metrics(metrics):
    from ems.info.joints import mmm_joints
    pose_names = mmm_joints[1:]
    dico = {key: val.numpy() for key, val in metrics.items()}

    if "APE_pose" in dico:
        APE_pose = dico.pop("APE_pose")
        for name, ape in zip(pose_names, APE_pose):
            dico[f"APE_pose_{name}"] = ape

    if "APE_joints" in dico:
        APE_joints = dico.pop("APE_joints")
        for name, ape in zip(mmm_joints, APE_joints):
            dico[f"APE_joints_{name}"] = ape

    if "AVE_pose" in dico:
        AVE_pose = dico.pop("AVE_pose")
        for name, ave in zip(pose_names, AVE_pose):
            dico[f"AVE_pose_{name}"] = ave

    if "AVE_joints" in dico:
        AVE_joints = dico.pop("AVE_joints")
        for name, ape in zip(mmm_joints, AVE_joints):
            dico[f"AVE_joints_{name}"] = ave

    return dico

@hydra.main(version_base=None, config_path="configs", config_name="sample_temos")
def _sample(cfg: DictConfig):
    return sample(cfg)


def cfg_mean_nsamples_resolution(cfg):
    if cfg.mean and cfg.number_of_samples > 1:
        logger.error("All the samples will be the mean.. cfg.number_of_samples=1 will be forced.")
        cfg.number_of_samples = 1

    return cfg.number_of_samples == 1


def get_path(sample_path: Path, is_amass: bool, gender: str, split: str, onesample: bool, mean: bool, fact: float):
    extra_str = ("_mean" if mean else "") if onesample else "_multi"
    fact_str = "" if fact == 1 else f"{fact}_"
    gender_str = gender + "_" if is_amass else ""
    path = sample_path / f"{fact_str}{gender_str}{split}{extra_str}"
    return path


def load_checkpoint(model, last_ckpt_path, *, eval_mode):
    # Load the last checkpoint
    # model = model.load_from_checkpoint(last_ckpt_path)
    # this will overide values
    # for example relative to rots2joints
    # So only load state dict is preferable
    import torch
    model.load_state_dict(torch.load(last_ckpt_path)["state_dict"])
    logger.info("Model weights restored.")

    if eval_mode:
        model.eval()
        logger.info("Model in eval mode.")


def sample(newcfg: DictConfig) -> None:
    # Load last config
    output_dir = Path(hydra.utils.to_absolute_path(newcfg.folder))
    last_ckpt_path = newcfg.last_ckpt_path

    # Load previous config
    prevcfg = OmegaConf.load(output_dir / ".hydra/config.yaml")
    # Overload it
    cfg = OmegaConf.merge(prevcfg, newcfg)
    onesample = cfg_mean_nsamples_resolution(cfg)

    logger.info("Sample script. The outputs will be stored in:")

    if "amass" in cfg.data.dataname:
        if "xyz" not in cfg.data.dataname:
            storage = output_dir / f"amass_samples_{cfg.jointstype}"
            assert "rots2joints" in cfg.transforms
            cfg.data.transforms.rots2joints.jointstype = cfg.jointstype
        else:
            if cfg.jointstype != "mmm":
                logger.info("This model has been trained with xyz joints, extracted from amass in the MMM 'format'.")
                logger.info("jointstype is then set to 'mmm'.")
            storage = output_dir / "amass_samples_mmm"
    else:
        storage = output_dir / "samples"

    path = get_path(storage, "amass" in cfg.data.dataname, cfg.gender, cfg.split, onesample, cfg.mean, cfg.fact)
    path.mkdir(exist_ok=True, parents=True)

    logger.info(f"{path}")

    import pytorch_lightning as pl
    import numpy as np
    import torch
    from hydra.utils import instantiate
    pl.seed_everything(cfg.seed)

    logger.info("Loading data module")
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    logger.info("Loading model")
    # Instantiate all modules specified in the configs

    if cfg.jointstype == "vertices":
        assert cfg.gender in ["male", "female", "neutral"]
        logger.info(f"The topology will be {cfg.gender}.")
        cfg.model.transforms.rots2joints.gender = cfg.gender

    model = instantiate(cfg.model,
                        nfeats=data_module.nfeats,
                        logger_name="none",
                        nvids_to_save=None,
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    # print(last_ckpt_path)
    load_checkpoint(model, last_ckpt_path, eval_mode=True)

    if "amass" in cfg.data.dataname and "xyz" not in cfg.data.dataname:
        model.transforms.rots2joints.jointstype = cfg.jointstype

    model.sample_mean = cfg.mean
    model.fact = cfg.fact

    if not model.hparams.vae and cfg.number_of_samples > 1:
        raise TypeError("Cannot get more than 1 sample if it is not a VAE.")

    from ems.data.tools.collate import collate_datastruct_and_text
    dataset = getattr(data_module, f"{cfg.split}_dataset")
    mse = torch.nn.MSELoss()
    from ems.data.sampling import upsample,subsample
    from rich.progress import Progress
    from rich.progress import track

    # remove printing for changing the seed
    logging.getLogger('pytorch_lightning.utilities.seed').setLevel(logging.WARNING)
    force_in_meter = cfg.jointstype != "mmmns"
    print("jointstype {}".format(cfg.jointstype))
    CMetrics = ComputeMetrics(force_in_meter=force_in_meter)
    import torch
    feat_save_dir = "/checkpoint/yijunq/release_feats"
    with torch.no_grad():
        with Progress(transient=True) as progress:
            task = progress.add_task("Sampling", total=len(dataset.keyids))
            for keyid in dataset.keyids:
                progress.update(task, description=f"Sampling {keyid}..")
                for index in range(cfg.number_of_samples):
                    one_data = dataset.load_eval_keyid(keyid)
                    # batch_size = 1 for reproductability
                    batch = collate_datastruct_and_text([one_data])
                    # fix the seed
                    pl.seed_everything(index)
                    features,connect_features = model.eval_forward(batch)
                    if batch["prev_text"][0] != -1:
                        connect_features = connect_features[:,model.temporal_window:]
                    if batch["next_text"][0] != -1:
                        connect_features = connect_features[:,:-model.temporal_window]
                    torch.save(connect_features.squeeze(0),os.path.join(feat_save_dir,keyid+".pt"))
                progress.update(task, advance=1)

    logger.info("All the sampling are done")
    logger.info(f"All the sampling are done. You can find them here:\n{path}")

if __name__ == '__main__':
    _sample()