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
                 framerate: float = 12.5,
                 progress_bar: bool = True,
                 pick_one_text: bool = True,
                 load_amass_data=False,
                 load_with_rot=False,
                 downsample=True,
                 add_noise = False,
                 noise_std = 0.0,
                 num_classes: int = 0,
                 nfeats = 135,
                 tiny: bool = False, **kwargs):

        self.split = split
        self.add_noise = add_noise
        self.noise_std = noise_std
        self.load_amass_data = load_amass_data
        self.load_with_rot = load_with_rot
        self.downsample = downsample
        self.num_classes = num_classes

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
            logger.info(f"There are {num_bad} sequences rejected by the sampler ({percentage:.4}%).")
    
    def _load_datastruct(self, keyid):
        features = torch.load(self.kitml_correspondances[keyid]["feat_path"])
        if self.add_noise:
            features+=torch.randn(features.size())*self.noise_std  
        datastruct = self.transforms.Datastruct(features=features)
        return datastruct
    
    def _load_eval_datastruct(self,keyid):
        gt_features = torch.load(self.kitml_correspondances[keyid]["feat_path"])
        pred_features = torch.load(self.kitml_correspondances[keyid]["gen_path"])
        gt_datastruct = self.transforms.Datastruct(features=gt_features)
        pred_datastruct = self.transforms.Datastruct(features=pred_features)
        return gt_datastruct,pred_datastruct

    def load_eval_keyid(self, keyid):
        labels = self.kitml_correspondances[keyid]["labels"][0]
        gt_datastruct,pred_datastruct = self._load_eval_datastruct(keyid)
        element = {"datastruct": gt_datastruct, "gen_datastruct": pred_datastruct,
                    "length": len(gt_datastruct), "keyids": keyid, "labels":labels}
        return element
    
    def __getitem__(self, index):
        keyid = self._split_index[index]
        return self.load_eval_keyid(keyid)

    def __len__(self):
        return len(self._split_index)

    def __repr__(self):
        return f"{self.dataname} dataset: ({len(self)}, _, ..)"