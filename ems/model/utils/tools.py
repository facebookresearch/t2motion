# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
def detach_to_numpy(tensor):
    return tensor.detach().cpu().numpy()


def remove_padding(tensors, lengths):
    return [tensor[:tensor_length] for tensor, tensor_length in zip(tensors, lengths)]
