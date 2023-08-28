# EMS: Elaborative Motion Synthesis
## Introduction
This is the official implementation of Breaking The Limits of Text-conditioned 3D Motion Synthesis with Elaborative Descriptions.

Please contact yijunq@meta.com if you have problem when using this code repo.
## Environmental Setup
### Install Packages 
Please use the environment.yaml file to install required packages.
```bash
    conda env create -f environment.yaml
```
### Download Body Model
Please register and download the SMPLH body model from [here](https://mano.is.tue.mpg.de/login.php), and follow the introductions in [TEMOS](https://github.com/Mathux/TEMOS#4-optional-smpl-body-model) to preprocess the files.
Finally, you should get a directory as below:
```  
${ROOT}  
|-- deps
|   |-- smplh
|   |   |-- SMPLH_FEMALE.npz
|   |   |-- SMPLH_MALE.npz
|   |   |-- SMPLH_NEUTRAL.npz
|   |   |-- smplh.faces
```
### Install Blender (Optional)
This is only needed if you want to render the 3D human figures.
We use blender to render the SMPL sequence, please install it from [here](https://www.blender.org/download/releases/2-93/). We build and test the code on blender 2.93, but higher version may also work.

## Data Preparation
Please download the annotation from [Here](), and SMPLH human motion data from [AMASS](https://amass.is.tue.mpg.de/). Currently, our model is trained with SMPLH body model, so please select the "SMPL+H G" icon in the download page.
After downloading, please change the "$path_to_amass" to where the AMASS dataset is downloaded and "$path_to_extracted_feature_folder" to the place where you want to store the extracted features.
Then run
```bash
cd util_tools
python preprocess.py --amass_path $path_to_amass --feat_folder $path_to_extracted_feature_folder
cd ../
```
You are expected to get the annotation file "babelsync.json" under "./datasets" and extracted feature files (xxx.pt) under your selected folder.

## Evaluation
To quickly evaluate our model, please firstly follow the data preparation steps to get the converted annotation file and feature folder, then download the pretrained model from [here](), and download the action recognition model from [here]().

To evaluate the model with APE&AVE metrics, simply run:
```bash
    python eval_temos.py folder=$path_to_pretrained_model_folder
```

To evaluate the model with Acc&FID metrics, you will take three steps:
Firstly, run
```bash
    python sample_clf.py folder=$path_to_pretrained_model_folder feat_save_dir=$path_to_sample_feat
```
to sample motion feature files with EMS.

Then, run
```bash
    cd util_tools
    python preprocess_clf.py --gt_feat_folder $path_to_extracted_feature_folder --feat_folder $path_to_sample_feat
    cd ../
```
to update the extracted ground truth feature path and generated feature path in the annotation file.

Finally, run
```bash
    python eval_clf.py folder=$path_to_action_recognition_model
```
to get the acc&fid metrics.

## Training
To train the EMS model yourself, please also follow the data preparation steps to get the converted annotation file and feature folder, then download the humor prior model from [here]() and place it under the "./outputs" folder.

Finally run the training script:
```bash
    python train.py data=babelsync-ems-mul-amass-rot run_id=iccv data.batch_size=8 model=ems model.optim.lr=5.0e-05 init_weight=/private/home/yijunq/repos/t2motion/outputs/humor_prior.pt
```

Our experiments are made on 8 V100 GPUs with a total batch size of 8X8=64, so you may need to change the optim.lr accordingly based on the GPUs you used.
## Interactive Rendering
To make it easier to use our model, we also provide an interactive code which takes in natural language descriptions and durations of each atomic action. Please modify the input information in "./input.json" then run (we provide several samples under "./texts"):
```bash
    python interact.py folder=./outputs/pretrained/ems

    blender --background --python render.py -- npy=./outputs/pretrained/ems/neutral_input/ems.npy mode=video
```
You are expected to get a ems.webv file under "./outputs/pretrained/ems/interact_samples/neutral_input".
## Acknowledgments
We want to especially thank the following contributors that our code is based on:
[TEMOS](https://github.com/Mathux/TEMOS),[MDM](https://github.com/GuyTevet/motion-diffusion-model), and [MultiAct](https://github.com/TaeryungLee/MultiAct_RELEASE/tree/main).
