#!/bin/bash
#SBATCH --job-name=run_vilbert_vqa    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/visual_bert/configs/vqa2/defaults.yaml \
        run_type=train_val \
        dataset=vqa2 \
        model=vilbert \
        checkpoint.resume_zoo=visual_bert.pretrained.coco.full \
        env.save_dir=save/visual_bert/vqa2 \
        env.data_dir=/fsx-labs/thwjoy/mmf/data

        



        
