#!/bin/bash
#SBATCH --job-name=run_visual_bert_visual_entailment    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_predict config=projects/visual_bert/configs/visual_entailment/defaults.yaml \
        run_type=test \
        dataset=visual_entailment \
        model=visual_bert \
        checkpoint.resume_file=$2 \
        env.save_dir=save/visual_bert/visual_entailment \
        env.data_dir=/fsx-labs/thwjoy/mmf/data

        



        
