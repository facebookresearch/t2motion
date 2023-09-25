# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
from glob import glob
from typing import Dict, Optional
import logging

import numpy as np
import pandas
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn.init import normal_
from pathlib import Path

from ems.tools.easyconvert import matrix_to, axis_angle_to
from ems.transforms import Transform
from ems.data.sampling import subsample
from ems.data.tools.smpl import smpl_data_to_matrix_and_trans

from rich.progress import track
        
from .base import BASEDataModule
from .utils import get_split_keyids,swap_left_right

logger = logging.getLogger(__name__)


class BABELDataModule(BASEDataModule):
    def __init__(self, data_dir: str = "",
                 batch_size: int = 32,
                 num_workers: int = 16,
                 **kwargs):
        super().__init__(batch_size=batch_size,
                         num_workers=num_workers)
        self.save_hyperparameters(logger=False)
        self.Dataset = BABEL

        sample_overrides = {"split": "train", "tiny": True,
                            "progress_bar": False}
        self._sample_set = self.get_sample_set(overrides=sample_overrides)

        # Get additional info of the dataset
        self.nfeats = self._sample_set.nfeats
        self.transforms = self._sample_set.transforms


class BABEL(Dataset):
    dataname = "BABEL"

    def __init__(self, datapath: str,
                 splitpath: str,
                 transforms: Transform,
                 split: str = "train",
                 transforms_xyz: Optional[Transform] = None,
                 transforms_smpl: Optional[Transform] = None,
                 correspondance_path: str = None,
                 sampler=None,
                 progress_bar: bool = True,
                 pick_one_text: bool = True,
                 load_amass_data=False,
                 load_with_rot=False,
                 downsample=True,
                 add_noise = False,
                 noise_std = 0.0,
                 temporal_window = 2,
                 nfeats = 135,
                 tiny: bool = False, **kwargs):

        self.split = split
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.load_amass_data = load_amass_data
        self.load_with_rot = load_with_rot
        self.downsample = downsample
        self.temporal_window = temporal_window
        self.temporal_pool = nn.AdaptiveAvgPool1d(temporal_window)

        if load_amass_data and not self.load_with_rot:
            self.transforms_xyz = transforms_xyz
            self.transforms_smpl = transforms_smpl
            self.transforms = transforms_xyz
        else:
            self.transforms = transforms

        self.sampler = sampler
        self.pick_one_text = pick_one_text

        super().__init__()
        keyids = get_split_keyids(path=splitpath, split=split)
        self.keyids = []

        if load_amass_data:
            with open(correspondance_path) as correspondance_path_file:
                self.kitml_correspondances = json.load(correspondance_path_file)

        if progress_bar:
            enumerator = enumerate(track(keyids, f"Loading BABEL {split}"))
        else:
            enumerator = enumerate(keyids)

        if tiny:
            maxdata = 4
        else:
            maxdata = np.inf

        datapath = Path(datapath)

        num_bad = 0

        for i, keyid in enumerator:
            if len(self.keyids) >= maxdata:
                break
            if keyid not in self.kitml_correspondances:
                num_bad += 1
                continue
            if not self.sampler.accept(self.kitml_correspondances[keyid]["num_frames"]):
                num_bad += 1
                continue
            self.keyids.append(keyid)
            # print(features.size())
            # e.g. (90,135)
            
        # self.keyids = list(features_data.keys())
        self._split_index = list(self.keyids)
        self.nfeats = nfeats
        if split != "test" and not tiny:
            total = len(self.keyids)
            percentage = 100 * num_bad / (total+num_bad)
            # logger.info(f"There are {num_bad} sequences rejected by the sampler ({percentage:.4}%).")
    
    def _load_contrast_datastruct(self,keyid):
        features = torch.load(self.kitml_correspondances[self.kitml_correspondances[keyid]["match_id"]]["feat_path"])
        return self.transforms.Datastruct(features=features)
    
    def _load_datastruct(self, keyid):
        features = torch.load(self.kitml_correspondances[keyid]["feat_path"])
        prev_keyid = self.kitml_correspondances[keyid]["prev"]
        next_keyid = self.kitml_correspondances[keyid]["next"]
        
        if prev_keyid != -1:
            prev_act_features = torch.load(self.kitml_correspondances[str(prev_keyid)]["feat_path"])
            if len(prev_act_features)<self.temporal_window:
                prev_features = self.temporal_pool(prev_act_features.unsqueeze(0).permute(0,2,1)).squeeze().permute(1,0)
            else:
                prev_features = prev_act_features[-self.temporal_window:]
        else:
            prev_features = torch.zeros(self.temporal_window,self.nfeats)
        
        if next_keyid != -1:
            next_act_features = torch.load(self.kitml_correspondances[str(next_keyid)]["feat_path"])
            if len(next_act_features)<self.temporal_window:
                next_features = self.temporal_pool(next_act_features.unsqueeze(0).permute(0,2,1)).squeeze().permute(1,0)
            else:
                next_features = next_act_features[:self.temporal_window]
        else:
            next_features = torch.zeros(self.temporal_window,self.nfeats)
        
        if self.add_noise:
            features+=torch.randn(features.size())*self.noise_std
            
        datastruct = self.transforms.Datastruct(features=features)
        prev_datastruct = self.transforms.Datastruct(features=prev_features)
        next_datastruct = self.transforms.Datastruct(features=next_features)
        connect_features = [features]
        if prev_keyid != -1:
            connect_features.insert(0,prev_features)
        if next_keyid != -1:
            connect_features.append(next_features)
        connect_datastruct = self.transforms.Datastruct(features=torch.cat(connect_features,dim=0))
        return datastruct,prev_datastruct,next_datastruct,connect_datastruct
        
    def _load_text(self, keyid):
        sequences = self.kitml_correspondances[keyid]["text"]
        if not self.pick_one_text:
            return sequences
        n = len(sequences)
        if self.split != "test":
            index = np.random.randint(n)
        else:
            # Only the first one in evaluation
            index = 0
        text = sequences[index]
        return text

    def load_keyid(self, keyid):
        text = self._load_text(keyid)
        if "match_id" not in self.kitml_correspondances[keyid]:
            self.kitml_correspondances[keyid]["match_id"] = str(int(keyid)-1)
        if self.kitml_correspondances[keyid]["match_id"] not in self.kitml_correspondances:
            self.kitml_correspondances[keyid]["match_id"] = self._split_index[self._split_index.index(keyid)-1]
        contrast_text = self._load_text(self.kitml_correspondances[keyid]["match_id"])
        
        if "labels" in self.kitml_correspondances[keyid]:
            labels = self.kitml_correspondances[keyid]["labels"][0]
        else:
            labels = 0
            
        datastruct,prev_datastruct,next_datastruct,connect_datastruct = self._load_datastruct(keyid)
        weight = self.kitml_correspondances[keyid]["avg_cnt"]
        contrast_datastruct = self._load_contrast_datastruct(keyid)
        element = {"datastruct": datastruct, "text": text, "next_datastruct": next_datastruct, "contrast_text":contrast_text, "contrast_datastruct":contrast_datastruct, "contrast_length": len(contrast_datastruct),
        "length": len(datastruct), "keyids": keyid, "prev_ids":self.kitml_correspondances[keyid]["prev"], "connect_datastruct":connect_datastruct,
        "prev_datastruct": prev_datastruct, "next_ids":self.kitml_correspondances[keyid]["next"], "labels":labels, "weights":weight}
       
        return element

    def load_eval_keyid(self, keyid):
        text = self._load_text(keyid)
        prev_id = self.kitml_correspondances[keyid]["prev"]
        next_id = self.kitml_correspondances[keyid]["next"]
        if prev_id != -1:
            prev_id = str(prev_id)
            prev_text = self._load_text(prev_id)
            prev_length = self.kitml_correspondances[prev_id]["num_frames"]
        else:
            prev_text = -1
            prev_length = -1
            
        if next_id != -1:
            next_id = str(next_id)
            next_text = self._load_text(next_id)
            next_length = self.kitml_correspondances[next_id]["num_frames"]
        else:
            next_text = -1
            next_length = -1
        labels = self.kitml_correspondances[keyid]["labels"]
        # labels = torch.zeros(self.num_classes)
        # for label in self.kitml_correspondances[keyid]["labels"]:
        #     labels[label] = 1
        datastruct,prev_datastruct,next_datastruct,connect_datastruct = self._load_datastruct(keyid)
        element = {"datastruct": datastruct, "text": text,"length": len(datastruct), "keyids": keyid, "prev_ids": prev_id, "next_ids": next_id,
                   "connect_datastruct": connect_datastruct, "labels": labels, "prev_text": prev_text, "next_text": next_text,
                   "prev_length":prev_length, "next_length":next_length}
        return element
    
    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_keyid(keyid)

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"