source $(conda info --base)/etc/profile.d/conda.sh
conda create --name gdrm_direct -y python=3.7.3
conda activate
conda activate gdrm_direct
conda install -y numpy=1.17.0 --force-reinstall
conda install -y pytorch=1.2.0 torchvision=0.4.0 cudatoolkit=10.0 -c pytorch --force-reinstall
conda install -y tensorboardX=1.8 --force-reinstall
conda install -y matplotlib=3.1.1 --force-reinstall
conda install -y scikit-image=0.15.0 --force-reinstall
conda install -y h5py=2.9.0 --force-reinstall
conda install -y pandas=0.25.1 --force-reinstall
pip install advertorch==0.2.0
