from typing import List, Dict
from torch import Tensor
import torch

def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def collate_datastruct_and_text(lst_elements: List) -> Dict:
    collate_datastruct = lst_elements[0]["datastruct"].transforms.collate
    batch = {
        # Collate with padding for the datastruct
        "datastruct": collate_datastruct([x["datastruct"] for x in lst_elements]),
        # Collate normally for the length
        "length": [x["length"] for x in lst_elements]
        # Collate the text
        }
    if "text" in lst_elements[0]:
        batch["text"] = [x["text"] for x in lst_elements]
    if "labels" in lst_elements[0]:
        # print([x["labels"] for x in lst_elements])
        batch["labels"] = torch.tensor([x["labels"] for x in lst_elements])
    if "weights" in lst_elements[0]:
        sumconts = sum([x["weights"] for x in lst_elements])
        rev_cont = [x["weights"]/sumconts for x in lst_elements]
        ratio = sum(rev_cont)/len(rev_cont)
        batch["weights"] = torch.tensor([x/ratio for x in rev_cont])
    if "prev_datastruct" in lst_elements[0]:
        batch["prev_datastruct"] = collate_datastruct([x["prev_datastruct"] for x in lst_elements])
    if "next_datastruct" in lst_elements[0]:
        batch["next_datastruct"] = collate_datastruct([x["next_datastruct"] for x in lst_elements])
    if "connect_datastruct" in lst_elements[0]:
        batch["connect_datastruct"] = collate_datastruct([x["connect_datastruct"] for x in lst_elements])
    if "contrast_datastruct" in lst_elements[0]:
        batch["contrast_datastruct"] = collate_datastruct([x["contrast_datastruct"] for x in lst_elements])
    if "gen_datastruct" in lst_elements[0]:
        batch["gen_datastruct"] = collate_datastruct([x["gen_datastruct"] for x in lst_elements])
    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]
    return batch

def collate_datastruct_and_post_datastruct(lst_elements: List) -> Dict:

    collate_datastruct = lst_elements[0]["datastruct"].transforms.collate
    batch = {
        # Collate with padding for the datastruct
        "datastruct": collate_datastruct([x["datastruct"] for x in lst_elements]),
        # Collate normally for the length
        "length": [x["length"] for x in lst_elements],
        "post_length": [x["post_length"] for x in lst_elements],
        # Collate the text
        "post_datastruct": collate_datastruct([x["post_datastruct"] for x in lst_elements])}
    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]
    return batch

def collate_text_and_length(lst_elements: Dict) -> Dict:
    batch = {"length": [x["length"] for x in lst_elements],
             "text": [x["text"] for x in lst_elements]}

    # add keyid for example
    otherkeys = [x for x in lst_elements[0].keys() if x not in batch and x != "datastruct"]
    for key in otherkeys:
        batch[key] = [x[key] for x in lst_elements]
    return batch
