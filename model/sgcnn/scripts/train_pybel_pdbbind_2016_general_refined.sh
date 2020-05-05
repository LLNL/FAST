#!/bin/bash

################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Example training script for Spatial Graph Convolution Network
################################################################################


source activate pytorch_geometric

cd ~/aha/gcnn/pdbbind_2007_geometric

timestamp=$(date +%b_%d_%y_%H_%M_%s)
experiment_name="pdbbind_2016_general_refined"

python train.py --checkpoint-dir=pybel_processed_${experiment_name}_${timestamp} --num-workers=8 --batch-size=8 --preprocessing-type=processed --feature-type=pybel --epochs=300 --lr=1e-3 --covalent-threshold=1.5 --non-covalent-threshold=4.5 --covalent-gather-width=16 --covalent-k=2 --non-covalent-gather-width=12 --non-covalent-k=2 --checkpoint=True --checkpoint-iter=100 --train-data /g/g13/jones289/data/pdbbind_2016_pybel_processed_train_val_test/general_train.hdf /g/g13/jones289/data/pdbbind_2016_pybel_processed_train_val_test/refined_train.hdf --val-data /g/g13/jones289/data/pdbbind_2016_pybel_processed_train_val_test/general_val.hdf /g/g13/jones289/data/pdbbind_2016_pybel_processed_train_val_test/refined_val.hdf --dataset-name pdbbind

