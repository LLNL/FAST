#!/usr/bin/bash
export PATH=/usr/workspace/wsa/jones289/miniconda3/bin:$PATH
source activate pytorch_geometric

# path to directory containing the source code
export PYTHONPATH=/p/lscratchh/jones289/aha:$PYTHONPATH

# load LC CUDA 10 installation
module load cuda/10.0.130

export CUDA_HOME=/usr/tce/packages/cuda/cuda-10.0.130
