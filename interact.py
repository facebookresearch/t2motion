import logging

import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import ems.launch.prepare  # noqa
import numpy as np
import json
import ems.tools.geometry as geometry

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="interact")
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

def preprocess_input(input_dict_path):
    input_dict = json.load(open(input_dict_path))
    # print(input_dict)
    output_dict = {}
    texts = input_dict["texts"]
    processed_text = []
    output_dict["lengths"] = input_dict["durations"]
    for i in range(len(texts)):
        if i == len(texts)-1:
            processed_text.append(prev+", "+texts[i]+".")
        elif i == 0:
            processed_text.append(texts[i]+", "+texts[i+1])
        else:
            processed_text.append(prev+", "+texts[i]+", "+texts[i+1])
        if "the person" in texts[i]:
            prev = texts[i].split("the person ")[-1]
        else:
            prev = texts[i]
    output_dict["texts"] = processed_text
    return output_dict

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

    storage = output_dir / "interact_samples"
    cfg.split = cfg.input_dict.split("/")[-1].split(".")[0]
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
    mse = torch.nn.MSELoss()
    from ems.data.sampling import upsample,subsample
    from rich.progress import Progress
    from rich.progress import track

    # remove printing for changing the seed
    logging.getLogger('pytorch_lightning.utilities.seed').setLevel(logging.WARNING)
    with torch.no_grad():
        input_dict = preprocess_input(cfg.input_dict)
        print(input_dict["texts"])
        seqs = []
        connects = []
        prev_feats = None
        print("Generating Atomic Actions")
        num_seqs = len(input_dict["texts"])
        for i in range(num_seqs):
            batch = {"text":[input_dict["texts"][i]],"length":[input_dict["lengths"][i]]}
            # print(batch["text"])
            # fix the seed
            pl.seed_everything(0)
            features = model(batch)
            seqs.append(features)
        motion = torch.cat(seqs,dim=1)
        motion = model.Datastruct(features=motion).joints.squeeze().numpy()    
        npypath = path / f"assemb.npy"
        np.save(npypath, motion)
            
        num_seqs = len(seqs)
        if num_seqs>1:
            with Progress(transient=True) as progress:
                task = progress.add_task("Connecting", total=num_seqs)
                for i in range(num_seqs):
                    for index in range(cfg.number_of_samples):
                        pl.seed_everything(index)
                        batch = {}
                        batch["cur_act"] = seqs[i]
                        if i-1>0:
                            if seqs[i-1].size(1)<model.temporal_window:
                                batch["prev_act"] = seqs[i-1]
                            else:
                                batch["prev_act"] = seqs[i-1][:,-model.temporal_window:]
                        if i+1 < num_seqs:
                            if seqs[i+1].size(1)<model.temporal_window:
                                batch["next_act"] = seqs[i+1]
                            else:
                                batch["next_act"] = seqs[i+1][:,:model.temporal_window]
                        connect_features = model(batch)                       
                        if "prev_act" in batch and "next_act" in batch:
                            connects.append(connect_features[:,batch["prev_act"].size(1):-batch["next_act"].size(1)])
                        elif "prev_act" in batch:
                            connects.append(connect_features[:,batch["prev_act"].size(1):])
                        elif "next_act" in batch:
                            connects.append(connect_features[:,:-batch["next_act"].size(1)])
                    progress.update(task, advance=1)  
        if len(connects):
            soft_connects = []
            for connect in connects:
                if not len(soft_connects):
                    soft_connects.append(connect)
                else:
                    lerp_frame = torch.lerp(soft_connects[-1][:,-1],connect[:,0],0.5).unsqueeze(1)
                    soft_connects.extend([lerp_frame,connect])
            
            motion = torch.cat(soft_connects,dim=1)
            num_frames = motion.size(1)
            motion_datastruct = model.Datastruct(features=motion)           
            motion = motion_datastruct.joints.squeeze().numpy()
            npypath = path / f"ems.npy"
            np.save(npypath, motion)    
        
    logger.info("All the sampling are done")
    logger.info(f"All the sampling are done. You can find them here:\n{path}")

if __name__ == '__main__':
    _sample()
