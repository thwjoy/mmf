#!/bin/bash
#SBATCH --job-name=run_visual_entailment    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --requeue

module load python

source ~/.bashrc
source activate mmf

mmf_run config=projects/xtransform/configs/visual_entailment.yaml \
        model=xgen \
        run_type=train \
        datasets=visual_entailment \
        checkpoint.resume_file=save/$1/best.ckpt \
        env.save_dir=save/$1/visual_entailment \
        training.batch_size=512 \
        env.data_dir=/fsx-labs/thwjoy/mmf/data \
        training.evaluation_interval=1000 \
        checkpoint.reset.optimizer=true 
        



        
