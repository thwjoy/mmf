#!/bin/bash
#SBATCH --job-name=run_hateful_memes    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --requeue

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs/hateful_memes.yaml \
        model=xgen \
        run_type=train \
        datasets=hateful_memes \
        env.save_dir=save/random/hateful_memes \
        training.batch_size=512 \
        env.data_dir=/fsx-labs/thwjoy/mmf/data \
        training.evaluation_interval=1000 \
        checkpoint.reset.optimizer=true 
        
        



        
