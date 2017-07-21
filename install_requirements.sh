#!/bin/bash
pip install tqdm

if [ "$1" = "gpu" ]; then
	pip install tensorflow-gpu
else 
	pip install tensorflow
fi
pip install librosa