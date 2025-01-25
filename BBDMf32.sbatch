#!/bin/bash
#
#SBATCH --job-name=BBDMf32
#SBATCH --output=BBDMf32.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=END
#SBATCH --mail-user=zl3958@nyu.edu

module load anaconda3/2024.02;
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh;
conda activate /scratch/zl3958/BBVFI/;
export PATH=/scratch/zl3958/BBVFI/bin:$PATH;
cd /scratch/zl3958/VLPR/BBVFI

python3 main.py --config configs/Template-LBBDM-video.yaml --train --save_top --gpu_ids 0 
#python3 main.py --config /scratch/zl3958/VLPR/BBDM/configs/Template-LBBDM-f32.yaml --train --save_top --gpu_ids 0 --resume_model 

#python3 main.py --config /scratch/zl3958/VLPR/BBDM/configs/Template-LBBDM-f32.yaml --sample_to_eval --gpu_ids 0 --resume_model /scratch/zl3958/VLPR/BBDM/results/latest_model_25.pth