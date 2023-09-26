# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import hydra
import os
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from ems.model.metrics import ClfMetrics
import ems.launch.prepare  # noqa
import numpy as np
import json
from ems.model.metrics.fid import calculate_frechet_distance

logger = logging.getLogger(__name__)

def save_metric(path, metrics):
    strings = yaml.dump(metrics, indent=4, sort_keys=False)
    with open(path, "w") as f:
        f.write(strings)

def sanitize(dico):
    dico = {key: "{:.5f}".format(float(val)) for key, val in dico.items()}
    return dico

@hydra.main(version_base=None, config_path="configs", config_name="eval_clf")
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

    model = instantiate(cfg.model,
                        nfeats=data_module.nfeats,
                        logger_name="none",
                        nvids_to_save=None,
                        _recursive_=False)
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    # print(last_ckpt_path)
    load_checkpoint(model, last_ckpt_path, eval_mode=True)

    from ems.data.tools.collate import collate_datastruct_and_text
    dataset = getattr(data_module, f"{cfg.split}_dataset")
    from ems.data.sampling import upsample,subsample
    from rich.progress import Progress
    from rich.progress import track

    # remove printing for changing the seed
    logging.getLogger('pytorch_lightning.utilities.seed').setLevel(logging.WARNING)
    force_in_meter = cfg.jointstype != "mmmns"
    print("jointstype {}".format(cfg.jointstype))
    CMetrics = ClfMetrics()
    motion_latents = []
    gt_latents = []
    import torch
    with torch.no_grad():
        with Progress(transient=True) as progress:
            task = progress.add_task("Sampling", total=len(dataset.keyids))
            for keyid in dataset.keyids:
                # print(keyid)
                progress.update(task, description=f"Sampling {keyid}..")
                for index in range(cfg.number_of_samples):
                    one_data = dataset.load_eval_keyid(keyid)
                    # batch_size = 1 for reproductability
                    batch = collate_datastruct_and_text([one_data])
                    # fix the seed
                    pl.seed_everything(index)
                    motion_latent,motion_probs,gt_latent,gt_probs = model.eval_forward(batch)
                    motion_latents.append(motion_latent)
                    gt_latents.append(gt_latent)
                    CMetrics.update(motion_probs,batch["labels"])
                progress.update(task, advance=1)
            print("FID:{}".format(calculate_frechet_distance(torch.cat(motion_latents,dim=0),torch.cat(gt_latents,dim=0))))
            metrics = sanitize(CMetrics.compute())
            for k,v in metrics.items():
                print("{}:{}".format(k,v))

    logger.info("All the sampling are done")

if __name__ == '__main__':
    _sample()
