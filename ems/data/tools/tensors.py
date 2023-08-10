from typing import List, Dict
import torch
from torch import Tensor


def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def mask_to_lengths(mask):
    lengths = torch.where(mask>0, torch.full_like(mask,1),mask)
    lengths = lengths.sum(dim=1).tolist()
    return lengths

def mae_to_mask(lengths: List[int], indices, device: torch.device) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    for i in range(len(indices)):
        indice = indices[i]
        for digit in indice:
            mask[i][digit] = False
    return mask
    