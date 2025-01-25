#!/bin/bash
#
#SBATCH --job-name=Compute_flow
#SBATCH --output=ComputeFlow.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=160:00:00 
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=END
#SBATCH --mail-user=zl3958@nyu.edu

module load anaconda3/2024.02;
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh;
conda activate /scratch/zl3958/BBVFI/;
export PATH=/scratch/zl3958/BBVFI/bin:$PATH;

cd /scratch/zl3958/VLPR/pytorch-liteflownet

python3 compute_flow_sept.py