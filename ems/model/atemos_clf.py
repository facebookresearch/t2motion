from importlib.metadata import distribution
from typing import List, Optional

import torch

from hydra.utils import instantiate

from torch import Tensor
from omegaconf import DictConfig
from ems.model.utils.tools import remove_padding
import torch.nn as nn

from ems.model.base import BaseModel
import torch.nn.functional as F


class TEMOSCLF(BaseModel):
    def __init__(self,
                 optim: DictConfig,
                 transforms: DictConfig,
                 nfeats: int,
                 vae: bool,
                 latent_dim: int,
                 discriminator: DictConfig = None,
                 if_clf: bool = True,
                 num_classes: int = 235,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.discriminator = instantiate(discriminator, nfeats=nfeats)
        self.if_clf = if_clf
        if self.if_clf:
            self.clf = nn.Sequential(
                    nn.Linear(latent_dim, num_classes),
                    nn.Sigmoid(),
                    nn.Dropout(p=0.5)
                )
        self.if_mask = False
        self.transforms = instantiate(transforms)
        self.Datastruct = self.transforms.Datastruct
        self.optimizer = instantiate(optim, params=self.parameters())
        
        self.__post_init__()

    # Forward: text => motion
    def forward(self, batch: dict) -> List[Tensor]:    
        pass
    
    def eval_forward(self, batch: dict) -> List[Tensor]:
        if self.if_clf:
            motion_latent = self.discriminator(batch["gen_datastruct"].rfeats,batch["length"])
            motion_probs = self.clf(motion_latent)
            gt_latent = self.discriminator(batch["datastruct"].rfeats,batch["length"])
            gt_probs = self.clf(gt_latent)
        else:
            motion_latent,motion_probs = self.discriminator.eval_forward(batch["gen_datastruct"].rfeats,batch["length"])
            gt_latent,gt_probs = self.discriminator.eval_forward(batch["datastruct"].rfeats,batch["length"])
        return motion_latent,motion_probs,gt_latent,gt_probs