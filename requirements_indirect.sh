source $(conda info --base)/etc/profile.d/conda.sh
conda create --name gdrm_indirect -y python=3.8.5
conda activate
conda activate gdrm_indirect
conda install -y numpy=1.19.5 --force-reinstall
conda install pytorch==1.7.1 torchvision==0.11.1 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
conda install -y tensorboardX=2.1 --force-reinstall
conda install -y matplotlib=3.3.2 --force-reinstall
conda install -y scikit-image=0.17.2 --force-reinstall
conda install -y h5py=2.10.0 --force-reinstall
conda install -y pandas=1.2.5 --force-reinstall
pip install advertorch==0.2.3
pip install git+https://github.com/RobustBench/robustbench.git@v1.0
pip install filelock==3.4.0
pip install pytorch-pretrained-biggan==0.1.1