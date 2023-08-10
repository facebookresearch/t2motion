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

        self.seqTransDecoder = nn.TransformerDecoder(seq_trans_decoder_layer,
                                                     num_layers=num_layers)

        self.final_layer = nn.Linear(latent_dim, output_feats)
        
    def forward(self, z: Tensor, features:Tensor, lengths: List[int], post_lengths: List[int]):
        mask = lengths_to_mask(post_lengths, z.device)
        mem_mask = lengths_to_mask(lengths,z.device)
        z_mask = torch.tensor([True]*len(lengths),device=z.device).view(-1,1)
        mem_mask = torch.cat([z_mask,mem_mask],dim=1)
        
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        nfeats = self.hparams.nfeats
        # print("latent distribution size: {}".format(z.size()))
        z = z[None]  # sequence of 1 element for the memory
        #[1,B,D]
        z = torch.cat([z,features.permute(1,0,2)],dim=0)
        #[1+T,B,D]
        
        # Construct time queries
        time_queries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        time_queries = self.sequence_pos_encoding(time_queries)
        # [T,B,D]

        # Pass through the transformer decoder
        # with the latent vector for memory
        # print(post_lengths,lengths)
        # print(time_queries.size(),z.size(),mask.size(),mem_mask.size())
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
