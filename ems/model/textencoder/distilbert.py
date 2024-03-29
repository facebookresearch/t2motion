# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List, Union
import pytorch_lightning as pl

import torch.nn as nn
import os

import torch
from torch import Tensor
from torch.distributions.distribution import Distribution


class DistilbertEncoderBase(pl.LightningModule):
    def __init__(self, modelpath: str,
                 finetune: bool = False,
                 textmodel: str = "distilbert") -> None:
        super().__init__()

        from transformers import AutoTokenizer, AutoModel
        from transformers import logging
        logging.set_verbosity_error()
        # Tokenizer
        self.textmodel = textmodel
        if self.textmodel == "distilbert":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.tokenizer = AutoTokenizer.from_pretrained(modelpath)

            # Text model
            self.text_model = AutoModel.from_pretrained(modelpath)
            # Don't train the model
            if not finetune:
                self.text_model.training = False
                for p in self.text_model.parameters():
                    p.requires_grad = False
                    
        elif self.textmodel == "T5":
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

            # Text model
            self.text_model = AutoModel.from_pretrained("google/flan-t5-base")
            # Don't train the model
            if not finetune:
                self.text_model.training = False
                for p in self.text_model.parameters():
                    p.requires_grad = False
                    
        # Then configure the model
        self.text_encoded_dim = self.text_model.config.hidden_size

    def train(self, mode: bool = True):
        self.training = mode
        for module in self.children():
            # Don't put the model in
            if module == self.text_model and not self.hparams.finetune:
                continue
            module.train(mode)
        return self

    def get_last_hidden_state(self, texts: List[str],
                              return_mask: bool = False
                              ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        encoded_inputs = self.tokenizer(texts, return_tensors="pt", padding=True)
        output = self.text_model(**encoded_inputs.to(self.text_model.device))
        if not return_mask:
            return output.last_hidden_state
        return output.last_hidden_state, encoded_inputs.attention_mask.to(dtype=bool)
