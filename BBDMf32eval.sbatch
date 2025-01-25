#!/bin/bash
#
#SBATCH --job-name=BBDMf32eval
#SBATCH --output=BBDMf32eval.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --mail-type=END
#SBATCH --mail-user=zl3958@nyu.edu

module load anaconda3/2024.02;
source /share/apps/anaconda3/2024.02/etc/profile.d/conda.sh;
conda activate /scratch/zl3958/BBVFI/;
export PATH=/scratch/zl3958/BBVFI/bin:$PATH;
cd /scratch/zl3958/VLPR/BBVFI

#python3 main.py --config configs/Template-LBBDM-video.yaml --gpu_ids 0 --sample_to_eval
python3 main.py --config configs/Template-LBBDM-video.yaml --gpu_ids 0 --resume_model /home/zonglin/BBVFI/results/VQGAN/vimeo_unet.pth --sample_to_eval

python3 batch_to_entire.py --latent --dataset FILM_hard --step 10

python3 copy_GT.py --latent --dataset FILM_hard

python3 eval.py --latent --dataset FILM_hard --step 10 

python3 batch_to_entire.py --latent --dataset Xiph_2K --step 10
 
python3 copy_GT.py --latent --dataset Xiph_2K

python3 eval.py --latent --dataset Xiph_2K --step 10 

  
#python3 evaluate.py --config configs/ldm/ldmvfi-vqflow-f32-c256-concat_max.yaml --ckpt  /scratch/zl3958/VLPR/benchmark/LDMVFI/ldmvfi-vqflow-f32-c256-concat_max.ckpt --metrics PSNR SSIM LPIPS --data_dir /scratch/zl3958/VLPR/data --out_dir eval_results/100/ --use_ddim

 