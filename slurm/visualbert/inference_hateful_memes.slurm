#!/bin/bash
#SBATCH --job-name=run_visual_bert_hateful_memes    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_predict config=projects/visual_bert/configs/hateful_memes/defaults.yaml \
        run_type=test \
        dataset=hateful_memes \
        model=visual_bert \
        checkpoint.resume_file=save/visual_bert/hateful_memes/best.ckpt \
        env.save_dir=save/visual_bert \
        env.data_dir=/fsx-labs/thwjoy/mmf/data

        



        
