#!/bin/bash
#SBATCH --job-name=run_uniter_hateful_memes    # Job name
#SBATCH --gres=gpu:8                  # Run on a single GPU
#SBATCH --output=serial_test_%j.log   # Standard output and error log

module load python

source ~/.bashrc
source activate mmf

mmf_predict config=projects/xtransform/configs_uniter/hateful_memes.yaml \
        run_type=val \
        dataset=hateful_memes \
        model=uniter \
        checkpoint.resume_file=save/uniter/hateful_memes/best.ckpt \
        env.save_dir=save/uniter/hateful_memes \
        env.data_dir=/fsx-labs/thwjoy/mmf/data

        



        
