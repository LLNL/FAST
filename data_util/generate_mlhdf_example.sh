#!/usr/bin/bash
################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# generate_mlhdf example script
################################################################################


export PATH=/g/g12/ahashare/.conda/envs/covid19/bin:/usr/gapps/bbs/TOSS-3/hdf_utils:.:$PATH
export PYTHONPATH=.
alias python="~/.conda/envs/covid19/bin/python"


INPUT_DIR="/p/lustre2/zhang30/PDBBIND/pdbbind_2019/scratch/dockHDF5"
OUTPUT_DIR="eval_mlhdfs"


python generate_mlhdf.py --hdfx-path hdfx --input-dir $INPUT_DIR --output-dir $OUTPUT_DIR --output-suffix _eval_ml.hdf
