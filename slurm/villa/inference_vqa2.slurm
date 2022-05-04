#!/bin/bash
#SBATCH --job-name=run_villa_vqa    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_predict config=projects/xtransform/configs_uniter/vqa2.yaml \
        run_type=val \
        dataset=vqa2 \
        model=uniter \
        checkpoint.resume_file=save/villa/vqa2/best.ckpt \
        env.save_dir=save/villa/vqa2 \
        env.data_dir=/fsx-labs/thwjoy/mmf/data

        



        
