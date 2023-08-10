import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl

from typing import List, Optional
from torch import nn, Tensor

from ems.model.utils import PositionalEncoding
from ems.data.tools import lengths_to_mask

class ActorAgnosticDecoder(pl.LightningModule):
    def __init__(self, nfeats: int,
                 latent_dim: int = 256, ff_size: int = 1024,
                 num_layers: int = 4, num_heads: int = 4,
                 dropout: float = 0.1,
                 activation: str = "gelu", **kwargs) -> None:

        super().__init__()
        self.save_hyperparameters(logger=False)

        output_feats = nfeats

        self.sequence_pos_encoding = PositionalEncoding(latent_dim, dropout)

        seq_trans_decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim,
                                                             nhead=num_heads,
                                                             dim_feedforward=ff_size,
                                                             dropout=dropout,
                                                             activation=activation)

        self.feat2latent = nn.Linear(nfeats, latent_dim)
        self.seqTransDecoder = nn.TransformerDecoder(seq_trans_decoder_layer,
                                                     num_layers=num_layers)

        self.final_layer = nn.Linear(latent_dim, output_feats)

    def forward(self, z: Tensor, prev_feats: Tensor, next_feats: Tensor, prev_ids, next_ids, lengths: List[int]):
        temporal_window_size = prev_feats.size(1)
        prev_latent = self.feat2latent(prev_feats)
        next_latent = self.feat2latent(next_feats)
        mask = lengths_to_mask(lengths, z.device)
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        nfeats = self.hparams.nfeats
        # print("latent distribution size: {}".format(z.size()))
        z = z[None]  # sequence of 1 element for the memory
        z = torch.cat([prev_latent.permute(1,0,2),z,next_latent.permute(1,0,2)],dim=0)
        mem_mask = torch.ones((z.size(1),z.size(0)),dtype=bool,device=z.device)
        for idx in range(len(prev_ids)):
            if prev_ids[idx] == -1:
                mem_mask[:,:temporal_window_size] = False
            if next_ids[idx] == -1:
                mem_mask[:,-temporal_window_size:] = False
        #[3+1+3,B,D]
        # Construct time queries
        time_queries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)
        # [T,B,D]

        # Pass through the transformer decoder
        # with the latent vector for memory
        output = self.seqTransDecoder(tgt=time_queries, memory=z,
                                      tgt_key_padding_mask=~mask,
                                      memory_key_padding_mask =~mem_mask)

        output = self.final_layer(output)
        # [T,B,D]
        # zero for padded area
        output[~mask.T] = 0
        # Pytorch Transformer: [Sequence, Batch size, ...]
        feats = output.permute(1, 0, 2)
        #[B,T,D]
        return feats
