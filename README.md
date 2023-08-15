# EMS: Elaborative Motion Synthesis
## Introduction
This is the official implementation of Breaking The Limits of Text-conditioned 3D Motion Synthesis with Elaborative Descriptions.

Please contact yijunq@meta.com if you have problem when using this code repo.
## Environmental Setup
Please use the environment.yaml file to install required packages.
```bash
    conda env create -f environment.yaml
```
We use blender to render the SMPL sequence, please install it from here[here](https://www.blender.org/download/releases/2-93/). We build and test the code on blender 2.93, but higher version may also work.
## Data Preparation
Please download the annotation from [BABEL](), and SMPLH human motion data from [AMASS](). 
## Evaluation
To quickly evaluate our model, please firstly follow the data preparation steps to get the converted annotation file and feature folder, then download the pretrained model from [here](), and download the action recognition model from [here]().

To evaluate the model with APE&AVE metrics, simply run:
```bash
    python sample_clf_eval.py split=eval folder=/private/home/yijunq/repos/text2motion/outputs/babelsync-clf-amass-rot/baseline/gru_clf
```
To evaluate the model with Acc&FID metrics, simply run:
```bash
    python sample_clf_eval.py split=eval folder=/private/home/yijunq/repos/text2motion/outputs/babelsync-clf-amass-rot/baseline/gru_clf
```
## Training
To train the EMS model yourself, please also follow the data preparation steps to get the converted annotation file and feature folder, then run the training script:
```bash
    python train.py data=babelsync-ems-mul-amass-rot run_id=iccv_submission model.if_weighted=true data.batch_size=8 model=ems model.if_humor=true model.if_bigen=false data.if_llm_aug=false model.if_uniq=false model.optim.lr=5.0e-05 model.latent_dim=256 model.losses.lmd_text2rfeats_recons=1.0 model.if_contrast=true model.if_physics=false init_weight=/private/home/yijunq/repos/t2motion/outputs/humor.pt
```
## Acknowledgments
We want to especially thank the following contributors that our code is based on:
[TEMOS](https://github.com/Mathux/TEMOS),[MDM](https://github.com/GuyTevet/motion-diffusion-model)
