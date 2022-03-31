conda create --name "$1" -y python=3.8 && \
conda activate "$1" && \
conda install -y pip && \
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html \
pip install -r requirements.txt
