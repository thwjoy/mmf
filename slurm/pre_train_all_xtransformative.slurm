#!/bin/bash
#SBATCH --job-name=run_xtransform    # Job name
#SBATCH --gres=gpu:1                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs/pre_train_xtran.yaml \
        model=xgen \
        datasets=coco,masked_conceptual_captions_12,masked_sbu,visual_genome \
        run_type=train \
        env.save_dir=$1/xtransform
