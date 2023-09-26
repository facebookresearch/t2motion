# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch


class KLLoss:
    def __init__(self):
        pass

    def __call__(self, q, p, weights=None):
        div = torch.distributions.kl_divergence(q, p)
        if weights is not None:
            weights = weights.view(-1,1)
            weights = weights.repeat(1,div.size(1))
            assert weights.size()==div.size()
            return (div*weights).mean()
        else:
            return div.mean()

    def __repr__(self):
        return "KLLoss()"


class KLLossMulti:
    def __init__(self):
        self.klloss = KLLoss()

    def __call__(self, qlist, plist):
        return sum([self.klloss(q, p)
                    for q, p in zip(qlist, plist)])

    def __repr__(self):
        return "KLLossMulti()"
