# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import List

import torch
from einops import rearrange
from torch import Tensor
from torchmetrics import Metric

from ems.transforms.joints2jfeats import Rifke
from ems.tools.geometry import matrix_of_angles
from ems.model.utils.tools import remove_padding


def l2_norm(x1, x2, dim):
    return torch.linalg.vector_norm(x1 - x2, ord=2, dim=dim)


def variance(x, T, dim):
    mean = x.mean(dim)
    out = (x - mean)**2
    out = out.sum(dim)
    return out / (T - 1)


class ClfMetrics(Metric):
    def __init__(self, dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        # ACC
        self.add_state("ACC1", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("ACC5", default=torch.tensor(0.), dist_reduce_fx="sum")

        # All metric
        self.metrics = ["ACC1","ACC5"]

    def compute(self):
        count = self.count
        metrics = {metric: getattr(self, metric) / count for metric in self.metrics}
        return {**metrics}

    def update(self, probs, labels):
        self.count += labels.size(0)
        preds = torch.argsort(probs,1)
        num_batch = preds.size(0)
        for i in range(num_batch):
            if preds[i,-1] == labels[i]:
                self.ACC1 += 1
                self.ACC5 += 1
            else:
                for j in range(2,6):
                    if preds[i,-j] == labels[i]:
                        self.ACC5 += 1
                        break
        return 

