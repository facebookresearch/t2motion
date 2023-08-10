#!/bin/bash
#SBATCH --job-name=ems-humor-bigen-0.6
#SBATCH --output=/checkpoint/%u/jobs/sample-%j.out
#SBATCH --error=/checkpoint/%u/jobs/sample-%j.err
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=80
#SBATCH --mem=500gb
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=yijunq@meta.com
#SBATCH --mail-type=end
#SBATCH --constraint=volta32gb
source /private/home/yijunq/anaconda3/bin/activate
conda activate temos
cd /private/home/yijunq/repos/t2motion
python train.py data=babelsync-ems-mul-amass-rot run_id=iccv_submission model.if_weighted=true data.batch_size=8 model=ems model.if_humor=true model.if_bigen=false data.if_llm_aug=false model.if_uniq=false model.optim.lr=5.0e-05 model.latent_dim=256 model.losses.lmd_text2rfeats_recons=1.0 model.if_contrast=true model.if_physics=false init_weight=/private/home/yijunq/repos/t2motion/outputs/humor.pt