#!/bin/bash
#SBATCH --job-name=run_vilbert_visual_entailment    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/visual_bert/configs/visual_entailment/defaults.yaml \
        run_type=train_val \
        dataset=visual_entailment \
        model=vilbert \
        checkpoint.resume_zoo=visuvilbertal_bert.pretrained.coco.full \
        env.save_dir=save/vilbert \
        env.data_dir=/fsx-labs/thwjoy/mmf/data

        



        
