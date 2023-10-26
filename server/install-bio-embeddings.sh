#!/bin/bash

# PLEASE NOTE: consider using virtual environment!

# install python3.7 (version 3.7 works best with the bio-embeddings package - couldn't make it work with 3.10/3.11)
sudo apt update && sudo apt upgrade
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install python3.7

# get pip for python3.7
sudo apt install python3.7-distutils
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.7 get-pip.py
sudo apt-get install python3.7-dev

# install bio-embeddings package using the borrowed GPU (that's why I use 'bitfusion' command)
bitfusion run -n 1 -p 0.25 -- python3.7 -m pip install biopython transformers bio-embeddings[transformers] --force-reinstall --upgrade
bitfusion run -n 1 -p 0.25 -- python3.7 -m pip install bio-embeddings[esm] --force-reinstall --upgrade
# bump the CUDA version in pyTorch 
bitfusion run -n 1 -p 0.25 -- python3.7 -m pip install torch==1.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html

export PATH=/home/skrhak/.local/bin:$PATH
export PYTHONPATH=/home/skrhak/.local/lib/python3.7/site-packages:$PYTHONPATH