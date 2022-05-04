#!/bin/bash
#SBATCH --job-name=run_hateful_memes    # Job name
#SBATCH --gres=gpu:1                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs/hateful_memes.yaml \
        model=xgen \
        run_type=train \
        datasets=hateful_memes \
        checkpoint.resume_file=save/$1/best.ckpt \
        env.save_dir=save/$1/hateful_memes \
        training.batch_size=512 \
        env.data_dir=/fsx-labs/thwjoy/mmf/data \
        checkpoint.reset.optimizer=true 
        



        
