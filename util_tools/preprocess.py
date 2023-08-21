import os
import torch
import numpy as np
import json
import sys
sys.path.append("..") 
from utils import load_amass_keyid
from ems.data.tools.smpl import smpl_data_to_matrix_and_trans
from ems.transforms.rots2rfeats.smplvelp import SMPLVelP

amass_path = "/datasets01/amass/121119"
amass_annt = "/private/home/yijunq/repos/t2motion/datasets/babelsync-ems-mul_amass_path_scratch.json"
feat_folder = "/checkpoint/yijunq/gt_feats"
feat_annt_path = "/private/home/yijunq/repos/t2motion/datasets/feat_amass_path.json"
norm_path = "/private/home/yijunq/repos/t2motion/deps/transforms/rots2rfeats/smplvelp/rot6d/babelsync-ems-amass-rot"
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