# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import argparse

parser = argparse.ArgumentParser(description="Add Gen Feature Path to Annotation")
parser.add_argument("--gt_feat_folder", type=str, default="/checkpoint/yijunq/gt_feats")
parser.add_argument('--feat_folder', type=str, default="/checkpoint/yijunq/iccv_last_feats")
parser.add_argument("--annt_path",type=str,default="../datasets/babelclf_ems.json")
parser.add_argument('--jname', type=str, default="../datasets/babelclf.json")
parser.add_argument("--val_lst",type=str,default="../datasets/babelsync-clf-splits/val")
args = parser.parse_args()

new_dict = {}
annt = json.load(open(args.annt_path,"r"))
val_lst = open(args.val_lst).readlines()
for line in val_lst:
    kid = line.strip()
    feat = kid+".pt"
    if not os.path.exists(os.path.join(args.gt_feat_folder,feat)):
        print("GT Motion File {} is Missing".format(feat))
        continue
    if not os.path.exists(os.path.join(args.feat_folder,feat)):
        print("GEN Motion File {} is Missing".format(feat))
        continue
    kid = feat.split(".")[0]
    new_dict[kid] = annt[kid]
    new_dict[kid]["gen_path"] = os.path.join(args.feat_folder,feat)
    new_dict[kid]["feat_path"] = os.path.join(args.gt_feat_folder,feat)
json_str = json.dumps(new_dict,indent=4)
with open(args.jname,'w') as save_json:
    save_json.write(json_str)