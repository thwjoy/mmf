#!/bin/bash
#SBATCH --job-name=run_visual_bert_hateful_memes    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/visual_bert/configs/hateful_memes/defaults.yaml \
        run_type=train_val \
        dataset=hateful_memes \
        model=visual_bert \
        checkpoint.resume_zoo=visual_bert.pretrained.coco.full \
        env.save_dir=save/visual_bert/hateful_memes \
        env.data_dir=/fsx-labs/thwjoy/mmf/data

        



        
