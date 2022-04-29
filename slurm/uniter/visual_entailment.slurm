#!/bin/bash
#SBATCH --job-name=run_uniter_visual_entailment    # Job name
#SBATCH --gres=gpu:1                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/uniter/configs/visual_entailment/defaults.yaml \
        run_type=train_val \
        dataset=visual_entailment \
        model=uniter \
        checkpoint.resume_zoo=uniter.pretrained \
        env.save_dir=$1/uniter \
        env.data_dir=/fsx-labs/thwjoy/mmf/data

        



        
