# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch


def reparameterize(mu, logvar, seed=None):
    std = torch.exp(logvar / 2)

    if seed is None:
        eps = std.data.new(std.size()).normal_()
    else:
        generator = torch.Generator(device=mu.device)
        generator.manual_seed(seed)
        eps = std.data.new(std.size()).normal_(generator=generator)

    return eps.mul(std).add_(mu)