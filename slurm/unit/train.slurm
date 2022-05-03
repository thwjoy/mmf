#!/bin/bash
#SBATCH --job-name=run_unit    # Job name
#SBATCH --gres=gpu:1                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

python mmf_cli/run.py     \
        config=projects/xtransform/configs_unit/all_3_datasets/shared_dec.yaml     \
        datasets=vqa2,visual_entailment,hateful_memes    \
        model=unit run_type=train  \
        checkpoint.resume_zoo=unit.coco.single_task \
        env.save_dir=save/unit \
        env.data_dir=/fsx-labs/thwjoy/mmf/data

        



        
