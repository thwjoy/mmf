#!/bin/bash
#SBATCH --job-name=run_visual_entailment    # Job name
#SBATCH --gres=gpu:1                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs/visual_entailment.yaml \
        model=xgen \
        datasets=visual_entailment \
        env.resume_file=save \
        env.save_dir=save/visual_entailment
        



        
