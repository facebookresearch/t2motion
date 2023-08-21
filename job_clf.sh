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
python train.py data=babelsync-clf-amass-rot run_id=gru_clf_allconsq_sub data.batch_size=8 model=atemos_clf model.optim.lr=5.0e-05