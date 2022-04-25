#!/bin/bash
#SBATCH --job-name=run_contrastive    # Job name
#SBATCH --gres=gpu:1                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs/vqa2.yaml \
        model=xgen \
        datasets=vqa2 \
        env.resume_file=$1/best.ckpt \
        env.save_dir=$1/vqa
        



        
