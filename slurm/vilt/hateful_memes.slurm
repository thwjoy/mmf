#!/bin/bash
#SBATCH --job-name=run_vilt_hm        # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs_vilt/hateful_memes/defaults.yaml \
       run_type=train_val \
       dataset=hateful_memes \
       model=vilt \
       env.save_dir=$1/vilt/hateful_memes \
       env.data_dir=/fsx-labs/thwjoy/mmf/data \
       training.find_unused_parameters=true