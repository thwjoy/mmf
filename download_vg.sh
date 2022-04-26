#!/bin/bash
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip -P $TORCH_HOME/datasets/visual_genome/image
wget https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip -P $TORCH_HOME/datasets/visual_genome/image
unzip $TORCH_HOME/datasets/visual_genome/image/images.zip -d $TORCH_HOME/datasets/visual_genome/image
unzip $TORCH_HOME/datasets/visual_genome/image/images2.zip -d $TORCH_HOME/datasets/visual_genome/image
rsync -a $TORCH_HOME/datasets/visual_genome/image/VG_100K_2/ $TORCH_HOME/datasets/visual_genome/image/VG_100K