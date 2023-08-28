import os
import sys

try:
    import bpy
    sys.path.append(os.path.dirname(bpy.data.filepath))
except ImportError:
    raise ImportError("Blender is not properly installed or not launch properly. See README.md to have instruction on how to install and use blender.")

import ems.launch.blender
import ems.launch.prepare  # noqa
import logging
from pathlib import Path
import hydra
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

def get_samples_folder(path, *, jointstype):
    output_dir = Path(hydra.utils.to_absolute_path(path))
    candidates = [x for x in os.listdir(output_dir) if "samples" in x]
    if not candidates:
        raise ValueError("There is no samples for this model.")

    amass = False
    for candidate in candidates:
        amass = amass or ("amass" in candidate)

    if amass:
        samples_path = output_dir / f"amass_samples_{jointstype}"
        if not samples_path.exists():
            jointstype = "mmm"
            samples_path = output_dir / f"amass_samples_mmm"
            if not samples_path.exists():
                raise ValueError("You must specify a correct jointstype.")
            logger.info(f"Samples from {jointstype} not found, take mmm instead.")
    else:
        samples_path = output_dir / "samples"
    return samples_path, amass, jointstype

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

@hydra.main(version_base=None, config_path="configs", config_name="render")
def _render_cli(cfg: DictConfig):
    return render_cli(cfg)


def extend_paths(path, keyids, *, onesample=True, number_of_samples=1):
    if not onesample:
        template_path = str(path / "KEYID_INDEX.npy")
        paths = [template_path.replace("INDEX", str(index)) for index in range(number_of_samples)]
    else:
        paths = [str(path / "KEYID.npy")]

    all_paths = []
    for path in paths:
        all_paths.extend([path.replace("KEYID", keyid) for keyid in keyids])
    return all_paths



def render_cli(cfg: DictConfig) -> None:
    if cfg.npy is None:
        if cfg.folder is None or cfg.split is None:
            raise ValueError("You should either use npy=XXX.npy, or folder=XXX and split=XXX")
        # only them can be rendered for now
        if not cfg.infolder:
            jointstype = cfg.jointstype
            assert ("mmm" in jointstype) or jointstype == "vertices"

        from ems.data.utils import get_split_keyids
        keyids = get_split_keyids(path=Path(cfg.path.datasets)/ "babelprev-splits", split=cfg.split)

        onesample = cfg_mean_nsamples_resolution(cfg)
        if not cfg.infolder:
            model_samples, amass, jointstype = get_samples_folder(cfg.folder,
                                                                  jointstype=cfg.jointstype)
            path = get_path(model_samples, amass, cfg.gender, cfg.split, onesample, cfg.mean, cfg.fact)
        else:
            path = Path(cfg.folder)

        paths = extend_paths(path, keyids, onesample=onesample, number_of_samples=cfg.number_of_samples)
    else:
        paths = [cfg.npy]

    from ems.render.blender import render
    from ems.render.video import Video
    import numpy as np

    init = True
    for path in paths:
        try:
            data = np.load(path)
        except FileNotFoundError:
            logger.info(f"{path} not found")
            continue

        if cfg.mode == "video":
            frames_folder = path.replace(".npy", "_frames")
        else:
            frames_folder = path.replace(".npy", ".png")

        if cfg.mode == "video":
            vid_path = path.replace(".npy", f".{cfg.vid_ext}")
            if os.path.exists(vid_path):
                continue

        out = render(data, frames_folder,
                     denoising=cfg.denoising,
                     oldrender=cfg.oldrender,
                     res=cfg.res,
                     canonicalize=cfg.canonicalize,
                     exact_frame=cfg.exact_frame,
                     num=cfg.num, mode=cfg.mode,
                     faces_path=cfg.faces_path,
                     downsample=cfg.downsample,
                     always_on_floor=cfg.always_on_floor,
                     init=init,
                     contacts = cfg.contacts,
                     gt=cfg.gt)

        init = False

        if cfg.mode == "video":
            if not os.path.exists(frames_folder):
                frames_folder+="_of"
            video = Video(frames_folder, fps=12.5, res=cfg.res)
            # if cfg.downsample:
            #     video = Video(frames_folder, fps=12.5, res=cfg.res)
            # else:
            #     video = Video(frames_folder, fps=100.0, res=cfg.res)

            video.save(out_path=vid_path)
            logger.info(vid_path)

        else:
            logger.info(f"Frame generated at: {out}")



if __name__ == '__main__':
    _render_cli()
