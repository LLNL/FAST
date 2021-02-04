################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Summarize all prediction results
################################################################################


import os
import sys
import argparse
import numpy as np
import pandas as pd
import csv
import h5py


parser = argparse.ArgumentParser()
parser.add_argument("--data-dir", default='')
parser.add_argument("--input-csv", default='')
parser.add_argument("--input-3dcnn-npy", default='')
parser.add_argument("--input-sgcnn-hdf", default='')
parser.add_argument("--input-fusion-npy", default='')
parser.add_argument("--output-csv", default='')
args = parser.parse_args()


data_dir = args.data_dir
input_csv_name = args.input_csv
input_3dcnn_npy_name = args.input_3dcnn_npy
input_sgcnn_hdf_name = args.input_sgcnn_hdf
input_fusion_npy_name = args.input_fusion_npy
output_csv_name = args.output_csv

#data_dir = '/Users/kim63/Desktop'
#input_csv_name = 'docking_eval_3dcnn.csv'
#input_3dcnn_npy_name = 'docking_eval_output_3dcnn_test_pred.npy'
#input_sgcnn_hdf_name = 'docking_eval_output_sgcnn.hdf'
#input_fusion_npy_name = ''
#output_csv_name = 'docking_eval_output_pred.csv'


input_3dcnn_npy = np.load(os.path.join(data_dir, input_3dcnn_npy_name))
input_sgcnn_hdf = h5py.File(os.path.join(data_dir, input_sgcnn_hdf_name), 'r')
if len(input_fusion_npy_name) > 0:
	input_fusion_npy = np.load(os.path.join(data_dir, input_fusion_npy_name))
else:
	input_fusion_npy = None

# read input csv to get ligand id
lig_list = []
pred_list = []
input_csv = pd.read_csv(os.path.join(data_dir, input_csv_name), sep=',')
for ind, x in enumerate(input_csv.values):
	lig_id = x[0]
	comp_id = "_".join(lig_id.split('_')[:-1])
	pose_id = lig_id.split('_')[-1]

	# 3dcnn result
	pred_3dcnn = input_3dcnn_npy[ind]
	
	# sgcnn result
	pred_sgcnn = ((input_sgcnn_hdf[comp_id][pose_id]).attrs['y_pred']).ravel()
	
	# fusion result
	if not input_fusion_npy == None:
		pred_fusion = input_fusion_npy[ind]
	else:
		pred_fusion = 0
	
	lig_list.append([comp_id, pose_id])
	pred_list.append([pred_3dcnn, pred_sgcnn[0], pred_fusion])

input_sgcnn_hdf.close()


# output csv
output_csv_fp = open(os.path.join(data_dir, output_csv_name), 'w')
output_csv_writer = csv.writer(output_csv_fp, delimiter=',')
output_csv_writer.writerow(['ComplexID', 'PoseID', '3DCNN', 'SGCNN', 'Fusion'])
for lig_id, pred in zip(lig_list, pred_list):
	output_csv_writer.writerow([lig_id[0], lig_id[1], pred[0], pred[1], pred[2]])
output_csv_fp.close()



