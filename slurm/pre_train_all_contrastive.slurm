#!/bin/bash
#SBATCH --job-name=run_contrastive    # Job name
#SBATCH -N 1
#SBATCH --gres=gpu:8                  # Run on a 8 GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --requeue

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs/pre_train_contrastive.yaml \
        model=xgen \
        datasets=coco \
        run_type=train \
        env.save_dir=save/contrastive_coco \
        env.data_dir=/fsx-labs/thwjoy/mmf/data \
        training.batch_size=512