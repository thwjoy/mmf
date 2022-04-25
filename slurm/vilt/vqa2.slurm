#!/bin/bash
#SBATCH --job-name=run_vilt_vqa2     # Job name
#SBATCH --gres=gpu:1                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

python mmf_run config=projects/xtransform/configs_vilt/vqa2/defaults.yaml \
       run_type=val \
       dataset=vqa2 \
       model=vilt
       env.save_dir=$1/hateful_memes