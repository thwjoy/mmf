#!/bin/bash
#SBATCH --job-name=run_visual_bert_vqa    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_predict config=projects/visual_bert/configs/vqa2/defaults.yaml \
        run_type=val \
        dataset=vqa2 \
        model=visual_bert \
        checkpoint.resume_file=save/visual_bert/vqa2/best.ckpt \
        env.save_dir=save/visual_bert/vqa2 \
        env.data_dir=/fsx-labs/thwjoy/mmf/data

        



        
