# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import argparse

parser = argparse.ArgumentParser(description="Select label over twenty for clf task")
parser.add_argument("--babel_path", type=str, default="/private/home/yijunq/datasets/babel_v1.0_release")
parser.add_argument("--amass_path_ann", type=str, default="../datasets/babelsync.json")
parser.add_argument("--jname", type=str, default="../datasets/babelclf_ems.json")
args = parser.parse_args()

cur_annt = json.load(open(args.amass_path_ann))
# babel_trn = json.load(open(os.path.join(args.babel_path,"train.json")))
babel_val = json.load(open(os.path.join(args.babel_path,"val.json")))

new_dict = {}
label_dict = {}

num_acts = 0
match_dict = {}

label_over_twenty = [
 'place',
 'turn',
 'walk',
 'take',
 'stand with arms down',
 'stand',
 'turn left',
 'bend down',
 'tpose',
 'step forward',
 'step backward',
 'sit down',
 'walk forward',
 'step backwards',
 'turn around',
 'look down',
 'stand still',
 'walk backwards',
 'stand in place',
 'turn right',
 'apose',
 'stand up',
 'pick up object',
 'walk back',
 'stumble',
 'put down object',
 'bend over',
 'sit',
 'knock',
 'step back',
 'turn back',
 'step up',
 'step down',
 'squat',
 'jump',
 'run',
 'walk backward',
 'kick',
 'pick up',
 'turn around to the left',
 'crouch',
 'jog',
 'pick something up with right hand',
 'set down',
 'throw',
 'catch',
 'place object',
 'catch the ball',
 'catch ball',
 'stop',
 'turn around left',
 'throw the ball',
 'right foot forward']

# for k,v in babel_trn.items():
#     feat_path = "/".join(v["feat_p"].split("/")[1:])
#     match_dict[feat_path] = []
#     try:
#         dur = v["dur"]
#         frame_labels = v["frame_ann"]["labels"]
#         for frame_label in frame_labels:
#             proc_label = frame_label["proc_label"]
#             if proc_label not in label_dict:
#                 label_dict[proc_label] = num_acts
#                 num_acts+=1
#             start = frame_label["start_t"]/dur
#             end = frame_label["end_t"]/dur
#             match_dict[feat_path].append({"label":proc_label,"dur":[start,end]})
#     except:
#         print("ANNT {} LOAD ERROR".format(k))

for k,v in babel_val.items():
    feat_path = "/".join(v["feat_p"].split("/")[1:])
    match_dict[feat_path] = []
    try:
        dur = v["dur"]
        frame_labels = v["frame_ann"]["labels"]
        for frame_label in frame_labels:
            proc_label = frame_label["proc_label"]
            if proc_label not in label_dict:
                label_dict[proc_label] = num_acts
                num_acts+=1
            start = frame_label["start_t"]/dur
            end = frame_label["end_t"]/dur
            match_dict[feat_path].append({"label":proc_label,"dur":[start,end]})
    except:
        print("ANNT {} LOAD ERROR".format(k))

f = open("../datasets/babelsync-clf-splits/val","w")
for k,v in cur_annt.items():
    trim_start = v["trim"]["start"]
    trim_end = v["trim"]["end"]
    path = v["path"]
    max_overlps = 0
    if path not in match_dict:
        continue
    infos = match_dict[path]
    max_ovlp = 0
    max_ovlp_id = -1
    for infoid in range(len(infos)):
        info = infos[infoid]
        try:
            ovlp = (min(trim_end,info["dur"][1])-max(trim_start,info["dur"][0]))/(trim_end-trim_start)
        except:
            print(cur_annt[k])
            print(k,trim_start,trim_end)
            raise RuntimeError("check org annt")
        if ovlp>max_ovlp:
            max_ovlp = ovlp
            max_ovlp_id = infoid
    if max_ovlp_id == -1:
        print("ORG ANNT {} MIS MATCH".format(k))
        continue
    if infos[max_ovlp_id]["label"] in label_over_twenty:
        new_dict[k] = v
        new_dict[k]["labels"] = [label_over_twenty.index(infos[max_ovlp_id]["label"])]
        f.write(k+"\n")
    
json_str = json.dumps(new_dict,indent=4)
with open(args.jname,'w') as save_json:
    save_json.write(json_str)

