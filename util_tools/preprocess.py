# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import numpy as np
import json
import sys
sys.path.append("..") 
from utils import load_amass_keyid
from ems.data.tools.smpl import smpl_data_to_matrix_and_trans
from ems.transforms.rots2rfeats.smplvelp import SMPLVelP

import argparse

parser = argparse.ArgumentParser(description="Add Gen Feature Path to Annotation")
parser.add_argument("--amass_path", type=str, default="/datasets01/amass/121119")
parser.add_argument("--amass_annt", type=str, default="../datasets/babelsync_ems.json")
parser.add_argument("--feat_folder", type=str, default="/checkpoint/yijunq/gt_feats")
parser.add_argument("--feat_annt_path", type=str, default="/private/home/yijunq/repos/t2motion/datasets/babelsync.json")
args = parser.parse_args()

amass_path = args.amass_path
amass_annt = args.amass_annt
feat_folder = args.feat_folder
feat_annt_path = args.feat_annt_path
norm_path = "./feat_norm"
annt_folder = ""

new_annt = {}

transform = SMPLVelP(path=norm_path,normalization=True,canonicalize=True,offset=True,rotation=False)
annt = json.load(open(amass_annt,"r"))

act_num_cnt = {}
for k,v in annt.items():
    labels = v["labels"]
    for label in labels:
        if label not in act_num_cnt:
            act_num_cnt[label] = 1
        else:
            act_num_cnt[label] += 1

print("NUM Annt Samples {}".format(len(annt)))
for keyid,v in annt.items():
    smpl_data, success, duration, trunc, prev_keyid, next_keyid = load_amass_keyid(keyid, amass_path,
    correspondances=annt, downsample=True, framerate=12.5)
    if not success:
        continue
    smpl_data = smpl_data_to_matrix_and_trans(smpl_data, nohands=True)
    features = transform(smpl_data)
    start = trunc[0]
    end = trunc[1]
    if os.path.exists(os.path.join(annt_folder,keyid+"_annotations.json")):
        text = json.load(open(os.path.join(annt_folder,keyid+"_annotations.json")))
    elif "text" in v:
        text = v["text"]
    else:
        continue
    new_annt[keyid] = v
    features_data = features[start:end]
    torch.save(features_data,os.path.join(feat_folder,keyid+".pt"))
    new_annt[keyid]["feat_path"] = os.path.join(feat_folder,keyid+".pt")
    new_annt[keyid]["num_frames"] = end-start+1
    new_annt[keyid]["text"] = text
    new_annt[keyid]["avg_cnt"] = 0 
    for label in new_annt[keyid]["labels"]:
        new_annt[keyid]["avg_cnt"] += act_num_cnt[label]
    new_annt[keyid]["avg_cnt"] /= len(new_annt[keyid]["labels"])
    # new_annt[keyid].pop("labels")

print("Found {} Samples".format(len(new_annt)))
json_str = json.dumps(new_annt,indent=4)
with open(feat_annt_path,'w') as save_json:
    save_json.write(json_str)