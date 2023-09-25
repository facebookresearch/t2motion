# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import argparse
import numpy as np

def if_hit(temp_a,temp_b):
    if min(temp_b[1],temp_a[1])>max(temp_b[0],temp_a[0]):
        return True
    else:
        return False

def sort_acts(label_dicts,num_frames,fps,duration):
    trims = [0]
    infos = []
    rets = []
    ratio  = int(fps/12.5)
    length_thresh = 200*ratio/num_frames*duration
    for label_dict in label_dicts:
        try:
            trims.append(label_dict["start_t"])
            trims.append(label_dict["end_t"])
            labels = []
            for label in label_dict["act_cat"]:
                if label not in labels:
                    labels.append(label)
            infos.append((label_dict["proc_label"],label_dict["start_t"],label_dict["end_t"],labels))
        except:
            print(label_dicts)
            return []
    # transfer "transition" to meaningful texts    
    infos = sorted(infos,key=lambda x:x[1])
    infos = [list(x) for x in infos]
    for i in range(1,len(infos)-1):
        if infos[i][0] == "transition":
            infos[i][0] = "transit from {} to {}".format(infos[i-1][0],infos[i+1][0])
    trims = list(set(trims))
    trims.sort()
    # print(length_thresh,num_frames,duration)
    # print(trims)
    # raise RuntimeError("check")
    new_trims = []
    for trim in trims:
        if not len(new_trims):
            new_trims.append(trim)
        else:
            if trim-new_trims[-1]<=length_thresh:
                new_trims.append(trim)
            else:
                num_seqs = (trim-new_trims[-1])//length_thresh + 1
                seq_length = (trim-new_trims[-1])/num_seqs
                inter_trim = new_trims[-1]+seq_length
                while inter_trim<trim:
                    new_trims.append(inter_trim)
                    inter_trim+=seq_length
                new_trims.append(trim)
    if len(trims) != len(new_trims):
        print(new_trims)
        print(trims)
    trims = new_trims
    trims.sort()
        
    for i in range(len(trims)-1):
        trim = [trims[i],trims[i+1]]
        ret = [trims[i],trims[i+1],[],""]
        for info in infos:
            if if_hit(trim,[info[1],info[2]]):
                if not len(ret[-1]):
                    ret[-1]+=info[0]
                else:
                    ret[-1] = ret[-1]+" and "+info[0]
                ret[2].extend(info[3])
        ret[2] = list(set(ret[2]))
        rets.append(ret)
    # ret [start_t,end_t,labels,text]
    return rets
    
parser = argparse.ArgumentParser(description="Prepare Babel Dataset")
parser.add_argument("--babel_path", type=str, default="/private/home/yijunq/datasets/babel_v1.0_release")
parser.add_argument("--split_path", type=str, default="../datasets/babelsync-ems-mul-splits")
parser.add_argument("--amass_path_ann", type=str, default="../datasets/babelsync_ems.json")
parser.add_argument("--amass_path",type=str,default="/datasets01/amass/121119")
args = parser.parse_args()

if not os.path.exists(args.split_path):
    os.makedirs(args.split_path)
# if not os.path.exists(args.out_path):
#     os.makedirs(args.out_path)
    
splits = ["train","val"]
amass_path_ann = {}
labels_cache = {}
num_labels = 0

atom_id = 1
for split in splits:
    #1. create _annotation.json
    #2. add path to babel_amass_path.json
    #3. add file to split list
    flist = []
    babel_ann = json.load(open(os.path.join(args.babel_path,"{}.json".format(split))))
    for k,v in babel_ann.items():
        try:
            if v["frame_ann"] is None or not len(v["frame_ann"]["labels"]):
                continue
        except:
            print(k,v)
            continue
        
        smpl_path = os.path.join(args.amass_path,"/".join(v["feat_p"].split("/")[1:]))
        if not os.path.exists(smpl_path):
            continue
        smpl_info = np.load(smpl_path)
        duration = v["dur"]
        frame_length = smpl_info["trans"].shape[0]
        frame_per_sec = frame_length/duration
        fps = smpl_info["mocap_framerate"].item()
        act_cats = sort_acts(v["frame_ann"]["labels"],frame_length,fps,duration)
        if not len(act_cats):
            continue
        else:
            if_mirror = False
            # try:
            if True:
                prev_script = "the person "
                for act_idx in range(len(act_cats)):
                    act_cat = act_cats[act_idx]
                    if act_idx == len(act_cats)-1:
                        next_script = "."
                    else:
                        next_script = ", "+act_cats[act_idx+1][-1]
                    [start_t,end_t,labels,script]=act_cat
                    if "left" in script or "right" in script:
                        if_mirror = True
                    start = start_t/duration
                    end = end_t/duration
                    flist.append(atom_id)
                    if prev_script == "the person ":
                        amass_path_ann[atom_id] = {
                            "path":"/".join(v["feat_p"].split("/")[1:]),
                            "trim":{"start":start,"end":end},"prev":-1,"next":atom_id+1,"labels":[]}
                    else:
                        amass_path_ann[atom_id] = {
                            "path":"/".join(v["feat_p"].split("/")[1:]),
                            "trim":{"start":start,"end":end},"prev":atom_id-1,"next":atom_id+1,"labels":[]}
                        if amass_path_ann[atom_id]["path"]!=amass_path_ann[atom_id-1]["path"]:
                            amass_path_ann[atom_id]["prev"] = -1
                        if atom_id-1 in amass_path_ann and amass_path_ann[atom_id-1]["path"]!=amass_path_ann[atom_id]["path"]:
                            amass_path_ann[atom_id-1]["next"] = -1
                    for label in labels:
                        if label == "transition" and amass_path_ann[atom_id]["prev"] != -1:
                            amass_path_ann[atom_id]["labels"].extend(amass_path_ann[amass_path_ann[atom_id]["prev"]]["labels"]) 
                        else:
                            if label in labels_cache:
                                amass_path_ann[atom_id]["labels"].append(labels_cache[label])
                            else:
                                labels_cache[label] = num_labels
                                amass_path_ann[atom_id]["labels"].append(labels_cache[label])
                                num_labels+=1
                    amass_path_ann[atom_id]["labels"] = list(set(amass_path_ann[atom_id]["labels"]))
                    # json_str = json.dumps([prev_script+script+next_script],indent=4)
                    amass_path_ann[atom_id]["text"] = [prev_script+script+next_script]
                    prev_script = script+", "
                    # with open(os.path.join(args.out_path,"{}_annotations.json".format(atom_id)), 'w') as save_json:
                    #     save_json.write(json_str)
                    atom_id+=1
                amass_path_ann[atom_id-1]["next"] = -1
            # except:
            else:
                print(k)
                continue

            try:
            # if True:
                prev_script = "the person "
                for act_idx in range(len(act_cats)):
                    act_cat = act_cats[act_idx]
                    if act_idx == len(act_cats)-1:
                        next_script = "."
                    else:
                        next_script = ", "+act_cats[act_idx+1][-1]
                    [start_t,end_t,labels,script]=act_cat
                    start = start_t/duration
                    end = end_t/duration
                    if if_mirror:
                        sync_script = script.replace("left","$#$").replace("right","left").replace("$#$","right")
                        sync_type = 0
                    else:
                        if np.random.uniform()<0.5:
                            sync_script = script+ " fast"
                            sync_type = 1
                        else:
                            sync_script = script+ " slowly"
                            sync_type = 2
                    flist.append(atom_id)

                    if prev_script == "the person ":
                        amass_path_ann[atom_id] = {
                            "path":"/".join(v["feat_p"].split("/")[1:]),
                            "trim":{"start":start,"end":end},"prev":-1,"next":atom_id+1,"sync":sync_type,"labels":[]}
                    else:
                        amass_path_ann[atom_id] = {
                            "path":"/".join(v["feat_p"].split("/")[1:]),
                            "trim":{"start":start,"end":end},"prev":atom_id-1,"next":atom_id+1,"sync":sync_type,"labels":[]}
                        if amass_path_ann[atom_id]["path"]!=amass_path_ann[atom_id-1]["path"]:
                            amass_path_ann[atom_id]["prev"] = -1
                        if atom_id-1 in amass_path_ann and amass_path_ann[atom_id-1]["path"]!=amass_path_ann[atom_id]["path"]:
                            amass_path_ann[atom_id-1]["next"] = -1
                    for label in labels:
                        if label == "transition" and amass_path_ann[atom_id]["prev"] != -1:
                            amass_path_ann[atom_id]["labels"].extend(amass_path_ann[amass_path_ann[atom_id]["prev"]]["labels"]) 
                        else:
                            if label in labels_cache:
                                amass_path_ann[atom_id]["labels"].append(labels_cache[label])
                            else:
                                labels_cache[label] = num_labels
                                amass_path_ann[atom_id]["labels"].append(labels_cache[label])
                                num_labels+=1
                    amass_path_ann[atom_id]["labels"] = list(set(amass_path_ann[atom_id]["labels"]))
                    # json_str = json.dumps([prev_script+sync_script+next_script],indent=4)
                    amass_path_ann[atom_id]["text"] = [prev_script+sync_script+next_script]
                    prev_script = sync_script+", "
                    # with open(os.path.join(args.out_path,"{}_annotations.json".format(atom_id)), 'w') as save_json:
                    #     save_json.write(json_str)
                    atom_id+=1
                amass_path_ann[atom_id-1]["next"] = -1
            except:
                print(k)
                continue

    with open(os.path.join(args.split_path,split),"w") as f:
        for fname in flist:
            f.write("{}\n".format(fname))
    f.close()

json_str = json.dumps(amass_path_ann,indent=4)
with open(args.amass_path_ann,'w') as save_json:
    save_json.write(json_str)

json_str = json.dumps(labels_cache,indent=4)
with open("./labels_dict.json",'w') as save_json:
    save_json.write(json_str)