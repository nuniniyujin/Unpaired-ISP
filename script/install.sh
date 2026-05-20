# Official docker from https://www.codabench.org/competitions/12932/#/pages-tab
#docker pull gosha20777/ntire-isp-scorer:v1


# My conda:
#conda create -n isp python=3.10
#conda activate isp
conda install numpy matplotlib
#conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
#conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia
conda install opencv
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install demosaicnet
pip install colour-demosaicing
pip install tensorboard
