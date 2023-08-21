import os
import json
import argparse

parser = argparse.ArgumentParser(description="Add Gen Feature Path to Annotation")
parser.add_argument("--gt_feat_folder", type=str, default="")
parser.add_argument('--feat_folder', type=str, default="")
parser.add_argument("--annt_path",type=str,default="../datasets/babelsync-ems-mul_amass_path_scratch.json")
parser.add_argument('--jname', type=str, default="../datasets/ems-mul-clf_amass_path.json")

args = parser.parse_args()

new_dict = {}
feats = os.listdir(args.feat_folder)
annt = json.load(open(args.annt_path,"r"))
for feat in feats:
    if not os.path.exists(os.path.join(args.gt_feat_folder,feat)):
        print("Motion File {} is Missing".format(feat))
        continue
    kid = feat.split(".")[0]
    new_dict[kid] = annt[kid]
    new_dict[kid]["gen_path"] = os.path.join(args.feat_folder,feat)
    new_dict[kid]["feat_path"] = os.path.join(args.gt_feat_folder,feat)
json_str = json.dumps(new_dict,indent=4)
with open(args.jname,'w') as save_json:
    save_json.write(json_str)