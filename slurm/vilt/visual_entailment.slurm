#!/bin/bash
#SBATCH --job-name=run_vilt_ve    # Job name
#SBATCH --gres=gpu:1                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

python mmf_run config=projects/xtransform/configs_vilt/visual_entailment/defaults.yaml \
       run_type=val \
       dataset=visual_entailment \
       model=vilt
       env.save_dir=$1/visual_entailment
        



        
