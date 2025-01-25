#!/bin/bash
#
#SBATCH --job-name=Autoencoder_new
#SBATCH --output=Autoencoder_new.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --time=160:00:00 
#SBATCH --gres=gpu:v100:2
#SBATCH --mail-type=END
#SBATCH --mail-user=zl3958@nyu.edu

module load anaconda3/2024.02;
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh;
conda activate /scratch/zl3958/BBVFI/;
export PATH=/scratch/zl3958/BBVFI/bin:$PATH;
cd /scratch/zl3958/VLPR/BBVFI

python3 Autoencoder/main.py --base configs/vqflow-f32.yaml -t --gpus 0,1 #--resume /scratch/zl3958/VLPR/BBVFI/logs/2024-08-21T23-23-46_vqflow-f32/checkpoints/epoch=000049.ckpt 