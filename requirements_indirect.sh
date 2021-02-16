source $(conda info --base)/etc/profile.d/conda.sh
conda create --name gdrm_indirect -y python=3.8.5
conda activate
conda activate gdrm_indirect
conda install -y numpy=1.19.1 --force-reinstall
conda install -y pytorch==1.7.0 torchvision==0.8.1 cudatoolkit=10.1 -c pytorch
conda install -y tensorboardX=2.1 --force-reinstall
conda install -y matplotlib=3.3.2 --force-reinstall
conda install -y scikit-image=0.17.2 --force-reinstall
conda install -y h5py=2.10.0 --force-reinstall
conda install -y pandas=1.2.0 --force-reinstall
pip install advertorch==0.2.3
