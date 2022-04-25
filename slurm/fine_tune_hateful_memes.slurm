#!/bin/bash
#SBATCH --job-name=run_hateful_memes    # Job name
#SBATCH --gres=gpu:1                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs/hateful_memes.yaml \
        model=xgen \
        datasets=hateful_memes \
        env.resume_file=$1 \
        env.save_dir=$1/hateful_memes
        



        
