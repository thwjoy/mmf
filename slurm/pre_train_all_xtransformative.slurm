#!/bin/bash
#SBATCH --job-name=run_xtransform    # Job name
#SBATCH -N 1
#SBATCH --gres=gpu:4                 # Run on a 8 GPU
#SBATCH --output=serial_test_%j.log  # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs/pre_train_xtran.yaml \
        model=xgen \
        datasets=coco,masked_sbu,masked_conceptual_captions_12,visual_genome \
        run_type=train \
        env.save_dir=save/xtransform \
        env.data_dir=/fsx-labs/thwjoy/mmf/data \
        training.batch_size=512
