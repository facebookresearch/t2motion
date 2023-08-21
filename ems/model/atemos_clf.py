from importlib.metadata import distribution
from typing import List, Optional

import torch

from hydra.utils import instantiate

from torch import Tensor
from omegaconf import DictConfig
from ems.model.utils.tools import remove_padding
import torch.nn as nn

from ems.model.metrics import ClfMetrics
from torchmetrics import MetricCollection
from ems.model.base import BaseModel
from torch.distributions.distribution import Distribution
from ems.data.tools import collate_tensor_with_padding,mask_to_lengths
import torch.nn.functional as F


class TEMOSCLF(BaseModel):
    def __init__(self,
                 losses: DictConfig,
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
        self.metrics = ClfMetrics()
        self.optimizer = instantiate(optim, params=self.parameters())
        
        self._losses = torch.nn.ModuleDict({split: instantiate(losses, vae=vae,
                                                               _recursive_=False)
                                            for split in ["losses_train", "losses_test", "losses_val"]})
        self.losses = {key: self._losses["losses_" + key] for key in ["train", "test", "val"]}
        
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
     
    def allsplit_step(self, split: str, batch, batch_idx):
        # Encode the text/decode to a motion
        # print(batch["keyids"])
        motion_latent = self.discriminator(batch["datastruct"].rfeats,batch["length"])
        if self.if_clf:
            motion_probs = self.clf(motion_latent)
        else:
            motion_probs = motion_latent
        loss = self.losses[split].update(motion_probs=motion_probs,gt_labels=batch["labels"])
        if loss is None:
            raise ValueError("Loss is None, this happend with torchmetrics > 0.7")
        
        if split == "val":
            # Compute the metrics
            self.metrics.update(motion_probs.detach(),
                                batch["labels"])
        return loss