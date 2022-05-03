#!/bin/bash
#SBATCH --job-name=run_villa_visual_entailment    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs_uniter/visual_entailment.yaml \
        run_type=train_val \
        dataset=visual_entailment \
        model=uniter \
        checkpoint.resume_zoo=villa.pretrained \
        env.save_dir=save/villa/visual_entailment \
        env.data_dir=/fsx-labs/thwjoy/mmf/data

        



        
