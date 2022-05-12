#!/bin/bash
#SBATCH --job-name=run_vilt_vqa2     # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_predict config=projects/xtransform/configs_vilt/vqa2/defaults.yaml \
       run_type=val \
       dataset=vqa2 \
       model=vilt \
       checkpint.resume_file=save/vilt/vqa2/best.ckpt \
       env.save_dir=save/vilt/vqa2 \
       env.log_dir=save/vilt/vqa2 \
       env.data_dir=/fsx-labs/thwjoy/mmf/data