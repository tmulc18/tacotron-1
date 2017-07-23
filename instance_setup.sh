#!/bin/bash
sudo apt-get install zip
sudo apt-get install unzip &
sudo apt-get install htop &


sudo apt-get update
sudo apt-get --assume-yes upgrade
sudo apt-get --assume-yes install tmux build-essential gcc g++ make binutils
sudo apt-get --assume-yes install software-properties-common

# install Anaconda for current user
mkdir downloads
cd downloads
wget "https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh" -O "Anaconda2-4.2.0-Linux-x86_64.sh"
bash "Anaconda2-4.2.0-Linux-x86_64.sh" -b

echo "export PATH=\"$HOME/anaconda2/bin:\$PATH\"" >> ~/.bashrc
export PATH="$HOME/anaconda2/bin:$PATH"

if [ "$1" == "gpu" ]; then
	# download and install GPU drivers
	wget "http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.44-1_amd64.deb" -O "cuda-repo-ubuntu1604_8.0.44-1_amd64.deb"

	sudo dpkg -i cuda-repo-ubuntu1604_8.0.44-1_amd64.deb
	sudo apt-get update
	sudo apt-get -y install cuda
	sudo modprobe nvidia
	nvidia-smi

	# CUDNN
	wget "http://files.fast.ai/files/cudnn.tgz" -O "cudnn.tgz"
	tar -zxf cudnn.tgz
	cd cuda
	sudo cp lib64/* /usr/local/cuda/lib64/
	sudo cp include/* /usr/local/cuda/include/

	echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64/" >>  ~/.bashrc
	export LD_LIBRARY_PATH=/usr/local/cuda/lib64/

	pip install tensorflow-gpu
else
	pip install tensorflow
fi
pip install tqdm &
pip install librosa &
pip install tensorboard &