#!/bin/bash

module add py-pip/21.3.1
export TMPDIR=$SCRATCHDIR

pip3 install keras_tuner --root /storage/plzen1/home/skrhakv/test_apolo
pip3 install tensorflow --root /storage/plzen1/home/skrhakv/test_apolo
pip3 install scikit-learn --root /storage/plzen1/home/skrhakv/test_apolo
pip3 install tensorflow-addons --root /storage/plzen1/home/skrhakv/test_apolo

export PATH=/storage/plzen1/home/skrhakv/test_apolo/cvmfs/software.metacentrum.cz/spack18/software/linux-debian11-x86_64_v2/gcc-10.2.1/python-3.9.12-rg2lpmkxpcq423gx5gmedbyam7eibwtc/bin:$PATH
export PYTHONPATH=/storage/plzen1/home/skrhakv/test_apolo/cvmfs/software.metacentrum.cz/spack18/software/linux-debian11-x86_64_v2/gcc-10.2.1/python-3.9.12-rg2lpmkxpcq423gx5gmedbyam7eibwtc/lib/python3.9/site-packages:$PYTHONPATH