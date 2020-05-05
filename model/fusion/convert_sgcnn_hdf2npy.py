################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Convert SG-CNN feature output hdf to numpy array
################################################################################


import sys
import argparse
import numpy as np
import h5py


parser = argparse.ArgumentParser()
parser.add_argument("--input-hdf", default='')
parser.add_argument("--output-npy", default='')
args = parser.parse_args()


input_hdf_path = args.input_hdf
output_npy_path = args.output_npy

#input_hdf_path = '/Users/kim63/Desktop/temp/fusion_test/eval_sgcnn.hdf'
#output_npy_path = '/Users/kim63/Desktop/temp/fusion_test/eval_sgcnn_feat.npy'



feat_list = []
pred_list = []

input_hdf = h5py.File(input_hdf_path, 'r')
for com_id in input_hdf.keys():
	input_com = input_hdf[com_id]
	for pose_id in range(0,11):  # assuming pose1 to pose10
		if not str(pose_id) in input_com:
			continue
			
		input_pose = input_com[str(pose_id)]
		feat = input_pose['hidden_features'][:].ravel()
		feat = feat[28:40]
		feat_list.append(feat)
		pred = input_pose.attrs['y_pred'].ravel()
		pred_list.append(pred)


np.save(output_npy_path, np.array(feat_list))
np.save(output_npy_path[:-8] + 'pred.npy', np.array(pred_list))
input_hdf.close()


