# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import hydra
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.plugins import DDPPlugin
import ems.launch.prepare  # noqa
import os
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="configs", config_name="train")
def _train(cfg: DictConfig):
    cfg.trainer.enable_progress_bar = True
    return train(cfg)

def train(cfg: DictConfig) -> None:
    working_dir = cfg.path.working_dir
    logger.info("Training script. The outputs will be stored in:")
    logger.info(f"{working_dir}")

    # Delayed imports to get faster parsing
    logger.info("Loading libraries")
    import torch
    import pytorch_lightning as pl
    from hydra.utils import instantiate
    from ems.logger import instantiate_logger
    logger.info("Libraries loaded")

    logger.info(f"Set the seed to {cfg.seed}")
    pl.seed_everything(cfg.seed)

    logger.info("Loading data module")
    data_module = instantiate(cfg.data)
    logger.info(f"Data module '{cfg.data.dataname}' loaded")

    logger.info("Loading model")
    print(cfg.model)
    model = instantiate(cfg.model,
                        nfeats=data_module.nfeats,
                        nvids_to_save=None,
                        _recursive_=False)
    
    if os.path.exists(cfg.init_weight):
        state_dict = torch.load(cfg.init_weight)["state_dict"]
        model_dict = model.state_dict()
        parse_dict = {}
        for k,v in state_dict.items():
            if "humor_encoder" in k:
                parse_dict[k] = v
            if "humor_decoder" in k:
                parse_dict[k] = v
        model_dict.update(parse_dict)
        model.load_state_dict(model_dict)
        num_freeze_params = 0
        for k,v in model.named_parameters():
            if k.startswith("humor"):
                num_freeze_params +=1
                v.requires_grad = False
        print("Init with weight {}".format(cfg.init_weight))
        print("Totally {} params frozen".format(num_freeze_params))
    logger.info(f"Model '{cfg.model.modelname}' loaded")
    logger.info("Loading callbacks")
    metric_monitor = {
        "Train_jf": "recons/text2jfeats/train",
        "Val_jf": "recons/text2jfeats/val",
        "Train_rf": "recons/text2rfeats/train",
        "Val_rf": "recons/text2rfeats/val",
        "Train_cf": "recons/connect2connect/train",
        "Val_cf": "recons/connect2connect/val",
        "Train_tf":"recons/rfeats2text/train",
        "Val_tf":"recons/rfeats2text/val",
        "Train_hf":"recons/text2humor/train",
        "Val_hf": "recons/text2humor/val",
        "Train_CLF": "clf/motion/train",
        "Val_CLF": "clf/motion/val",
        "Train_posclf":"clf/gt_clf/train",
        "Train_negclf":"clf/pred_clf/train",
        "Val_posclf":"clf/gt_clf/val",
        "Val_negclf":"clf/pred_clf/val",
        "APE root": "Metrics/APE_root",
        "APE mean pose": "Metrics/APE_mean_pose",
        "AVE root": "Metrics/AVE_root",
        "AVE mean pose": "Metrics/AVE_mean_pose",
        "ACC1": "Metrics/ACC1",
        "ACC5": "Metrics/ACC5"
    }

    callbacks = [
        pl.callbacks.RichProgressBar(),
        instantiate(cfg.callback.progress, metric_monitor=metric_monitor),
        instantiate(cfg.callback.latest_ckpt),
        instantiate(cfg.callback.last_ckpt)
    ]

    logger.info("Callbacks initialized")

    logger.info("Loading trainer")
    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        logger=None,
        callbacks=callbacks
    )
    logger.info("Trainer initialized")

    logger.info("Fitting the model..")
    if os.path.exists(os.path.join(cfg.path.working_dir,"checkpoints/last.ckpt")):
        print("resume from last epoch")
        trainer.fit(model, datamodule=data_module, ckpt_path=os.path.join(cfg.path.working_dir,"checkpoints/last.ckpt"))
    else:
        trainer.fit(model, datamodule=data_module)
    logger.info("Fitting done")

    checkpoint_folder = trainer.checkpoint_callback.dirpath
    logger.info(f"The checkpoints are stored in {checkpoint_folder}")
    logger.info(f"Training done. The outputs of this experiment are stored in:\n{working_dir}")


if __name__ == '__main__':
    _train()
