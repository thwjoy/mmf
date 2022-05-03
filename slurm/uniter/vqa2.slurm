#!/bin/bash
#SBATCH --job-name=run_uniter_vqa    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs_uniter/vqa2.yaml \
        run_type=train_val \
        dataset=vqa2 \
        model=uniter \
        checkpoint.resume_zoo=uniter.pretrained \
        env.save_dir=$1/uniter/vqa2 \
        env.data_dir=/fsx-labs/thwjoy/mmf/data

        



        
