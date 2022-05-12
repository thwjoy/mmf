#!/bin/bash
#SBATCH --job-name=run_vilt_ve    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_predict config=projects/xtransform/configs_vilt/visual_entailment/defaults.yaml \
       run_type=val \
       dataset=visual_entailment \
       model=vilt \
       checkpint.resume_file=save/vilt/visual_entailment/best.ckpt \
       env.save_dir=save/vilt/visual_entailment \
       env.log_dir=save/vilt/visual_entailment \
       env.data_dir=/fsx-labs/thwjoy/mmf/data
        



        
