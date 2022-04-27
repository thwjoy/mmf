#!/bin/bash
#SBATCH --job-name=run_contrastive    # Job name
#SBATCH --gres=gpu:4                  # Run on a 8 GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs/pre_train_contrastive.yaml \
        model=xgen \
        datasets=coco,masked_sbu,masked_conceptual_captions_12,visual_genome \
        run_type=train \
        env.save_dir=$1/contrastive \
        env.data_dir=/fsx-labs/thwjoy/mmf/data \
        training.batch_size=512