#!/usr/bin/bash
################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# example script to process pdbbind dataset
################################################################################


#SBATCH -t 1-00:00:00
#SBATCH -p pbatch

set -u

shopt -s globstar


num_jobs=$1

timestamp=$(date +%b_%d_%y_%H_%M_%e)


echo "using ${num_jobs} workers.."

parallel -j0 --timeout 600 --delay 2.5 --joblog prepare_chimera_pdbbind_2018_refined_docking_${timestamp}.out.test ./prepare_complexes_chimera.sh {} ::: find docking_parse_pipeline_test_run/**/*_pocket.pdb
echo "done."
