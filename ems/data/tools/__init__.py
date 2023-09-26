# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .tensors import lengths_to_mask, mae_to_mask, mask_to_lengths
from .collate import collate_text_and_length, collate_datastruct_and_text, collate_tensor_with_padding, collate_datastruct_and_post_datastruct
