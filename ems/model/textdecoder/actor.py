# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional
from torch import nn, Tensor

from ems.model.utils import PositionalEncoding
from ems.data.tools import lengths_to_mask


class TextDecoder(pl.LightningModule):
    def __init__(self, text_encoded_dim: int = 768,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        self.text_encoded_dim = text_encoded_dim

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.seqTransDecoder = nn.TransformerDecoder(seq_trans_decoder_layer,
                                                     num_layers=num_layers)

        self.final_layer = nn.Linear(latent_dim, self.text_encoded_dim)

    def forward(self, z: Tensor, lengths: List[int]):
        mask = lengths_to_mask(lengths, z.device)
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        # print("latent distribution size: {}".format(z.size()))
        
        z = z[None]  # sequence of 1 element for the memory
        #[B,D]->[1,B,D]
        # Construct time queries
        time_queries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)
        # [T,B,D]

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=time_queries, memory=z,
                                      tgt_key_padding_mask=~mask)

        output = self.final_layer(output)
        # [T,B,D]
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        #[B,T,D]
        return feats
