#!/bin/bash
#SBATCH --job-name=run_xtransform    # Job name
#SBATCH -N 1
#SBATCH --gres=gpu:8                 # Run on a 8 GPU
#SBATCH --output=serial_test_%j.log  # Standard output and error log
#SBATCH --requeue

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs/pre_train_xtran.yaml \
        model=xgen \
        datasets=coco \
        run_type=train \
        env.save_dir=save/xtransform_coco \
        env.data_dir=/fsx-labs/thwjoy/mmf/data \
        training.batch_size=512 \
        training.evaluation_interval=1000 \
