# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional, Union
from torch import nn, Tensor
from torch.distributions.distribution import Distribution

from ems.model.utils import PositionalEncoding
from ems.data.tools import lengths_to_mask, mae_to_mask

class MotionDiscriminator(pl.LightningModule):
    def __init__(self, nfeats: int, vae: bool,
                 latent_dim: int = 256, ff_size: int = 1024,
                 dropout: float = 0.1, num_layers: int = 4, num_classes: int = 53,
                 activation: str = "gelu", **kwargs) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        input_feats = nfeats
        self.input_size = latent_dim
        self.hidden_size = latent_dim
        self.hidden_layer = num_layers
        
        self.skel_embedding = nn.Linear(input_feats, latent_dim)

        self.recurrent = nn.GRU(latent_dim, latent_dim, num_layers)
        self.linear1 = nn.Linear(latent_dim, 30)
        self.linear2 = nn.Linear(30, num_classes)
        self.xyz = 3.0
        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)


    def initHidden(self, num_samples, layer, device):
        return torch.randn(layer, num_samples, self.hidden_size, device=device, requires_grad=False)

    def forward(self, features: Tensor, lengths: Optional[List[int]] = None, indices=None, hidden_unit=None) -> Union[Tensor, Distribution]:
        pass
    
    def eval_forward(self, features: Tensor, lengths: Optional[List[int]] = None, indices=None, hidden_unit=None) -> Union[Tensor, Distribution]:
        if lengths is None:
            lengths = [len(feature) for feature in features]

        device = features.device

        bs, nframes, nfeats = features.shape
        if indices is None:
            mask = lengths_to_mask(lengths, device)
        else:
            mask = mae_to_mask(lengths,indices,device)

        x = features
        # Embed each human poses into latent vectors
        x = self.skel_embedding(x)

        # Switch sequence and batch_size because the input of
        # Pytorch Transformer is [Sequence, Batch size, ...]
        x = x.permute(1, 0, 2)  # now it is [nframes, bs, latent_dim]
        # 46,8,256
        # Each batch has its own set of tokens
        if hidden_unit is None:
            # motion sequence: batch * length * (joint*3)
            hidden_unit = self.initHidden(x.size(1), self.hidden_layer, device=device)
        gru_o, _ = self.recurrent(x.float(), hidden_unit)
        lin1 = self.linear1(gru_o[-1, :, :]) 
        lin1 = torch.tanh(lin1)
        # dim (num_samples, output_size)
        lin2 = self.linear2(lin1)
        return lin1*self.xyz,lin2