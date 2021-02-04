################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# 3D CNN and modified point-net/sentence-net for binding affinity prediction
################################################################################


from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import argparse
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import h5py
import datetime

from data_reader import *
from dnn_general import *



# feature type: 4 types in hdf5 file
g_feat_tool_list = ['pybel', 'rdkit']
g_feat_tool_ind = 0

g_feat_type_list = ['raw', 'processed']
g_feat_type_ind = 1

g_feat_pdbbind_type_list = ['crystal', 'docking'] # for display
g_feat_pdbbind_type_list2 = ['pdbbind', 'docking'] # for reading hdf (not used)
g_feat_pdbbind_type_ind = 0

# model selection: 3DCNN vs sentence_net
g_model_type_ind = 0 # 0: 3dcnn, 1: sentence-net

g_model_3dcnn_type_list = ['model_3dcnn_simple', 'model_3dcnn_res', 'model_3dcnn_res2', 'model_3dcnn_res3', 'model_3dcnn_res4', 'model_3dcnn_atomnet']
g_model_3dcnn_type_ind = 1

g_model_snet_type_list = ['model_snet']
g_model_snet_type_ind = 0

g_csv_header = ['ligand_id', 'file_prefix', 'label', 'train_test_split']




# default parameters (can be modified by user arguments)
#g_main_dir = "../../data"
#g_main_dir = "/ds/data/pdbbind_3dcnn"
g_main_dir = "/p/gpfs1/kim63/covid19/pdbbind_3dcnn"

g_target_dataset = "pdbbind2016"
g_target_trainval_type = "refined"

# for testing other external dataset (g_run_mode=4, 5)
#g_external_dir = "/ds/data/pdbbind_3dcnn/relative_free_energy_v3"
#g_external_csv_file = "pybel_processed_crystal_48_radius1_sigma1_rot0_info.csv"
#g_external_hd_files = ["", "", "pybel_processed_crystal_48_radius1_sigma1_rot0_test.hdf"]
#g_external_dir = "/ds/data/pdbbind_3dcnn"
#g_external_csv_file = "pdbbind2016_general_pybel_processed_crystal_48_radius1_sigma1_rot0_info.csv"
#g_external_hd_files = ["pdbbind2016_general_pybel_processed_crystal_48_radius1_sigma1_rot0_train.hdf", "pdbbind2016_general_pybel_processed_crystal_48_radius1_sigma1_rot0_val.hdf", "pdbbind2016_general_pybel_processed_crystal_48_radius1_sigma1_rot0_test.hdf"]
#g_external_dir = "../../data"
#g_external_csv_file = "pybel_processed_crystal_48_48_radius1_sigma1_rot0_info.csv"
#g_external_hd_files = ["", "", "pybel_processed_crystal_48_48_radius1_sigma1_rot0_test.hdf"]

g_external_dir = "/p/gpfs1/kim63/covid19/output_hdf/protease2_AllApprovedDrugs_2018_0"
g_external_csv_file = "docking_eval_3dcnn.csv"
g_external_hd_files = ["", "", "docking_eval_3dcnn.hdf"]
g_external_hd_type = 1 # 1: 3dcnn, 2: general ML-HDF format (atom list - docking), 3: general ML-HDF format (atom list - crystal)


g_feat_suffix = ""
g_3D_suffix = ""

g_input_3D_relative_size = False
g_input_3D_size_angstrom = 48
g_input_3D_dim = 48  # 24, 32, 48
g_input_3D_atom_radius = 1
g_input_3D_atom_radii = False
g_input_3D_sigma = 1
g_input_3D_rotate = False  # above one can be applied only if this is true

# input and output parameters
g_csv_suffix = "info.csv"
g_csv_ind_input = 1
g_csv_ind_output = 2
g_csv_ind_split = 3

g_input_train_hd_suffix = "train.hdf"
g_input_val_hd_suffix = "val.hdf"
g_input_test_hd_suffix = "test.hdf"
g_input_hds = [None, None, None]
g_input_mode = 1 # 1: read 3d_train/3d_test hd files, 2: read feature hd files

g_input_feat_size = [19, 75]
g_input_dim = [0, 0, 0, 0]
g_input_type = 0
g_output_dim = [1]
g_output_type = 10

# training and other hyperparameters
g_run_mode = 1 # 1: training, 2: testing, 3: save features, 4: test external dataset, 5: save features of external dataset
g_epoch_count = 200 # 100
g_batch_size = 50 # 50
g_online_batch_size = 0 # 20
g_save_rate = 0 #20, 0-> only save when the test loss is lower than before
g_verbose = 2
g_val_each_epoch = True
g_test_save_output = True
g_optimizer_info = [1, 0.0007, 0.9, 0.999, 1e-08] # default is 0.0007
g_decay_info = [1, 100, 0.95]
g_loss_info = [2, 0, 0, 5e-2]  # 1: l1, 2: l2
g_model_subdir = ""



def get_feat_suffix():
	global g_target_dataset
	global g_target_trainval_type
	global g_feat_tool_list
	global g_feat_tool_ind
	global g_feat_type_list
	global g_feat_type_ind
	global g_feat_pdbbind_type_list
	global g_feat_pdbbind_type_ind

	feat_suffix = "%s_%s_%s_%s_%s" % (g_target_dataset, g_target_trainval_type, g_feat_tool_list[g_feat_tool_ind], g_feat_type_list[g_feat_type_ind], g_feat_pdbbind_type_list[g_feat_pdbbind_type_ind])
	return feat_suffix

def get_3D_suffix():
	global g_input_3D_atom_radii
	global g_input_3D_dim
	global g_input_3D_atom_radius
	global g_input_3D_sigma
	global g_input_3D_rotate

	if g_input_3D_atom_radii:
		d_suffix = "%d_radii_sigma%d_rot%d" % (g_input_3D_dim, g_input_3D_sigma, g_input_3D_rotate)
	else:
		d_suffix = "%d_radius%d_sigma%d_rot%d" % (g_input_3D_dim, g_input_3D_atom_radius, g_input_3D_sigma, g_input_3D_rotate)
	return d_suffix

def get_input_dim():
	global g_model_type_ind
	global g_input_3D_dim
	global g_input_feat_size
	global g_feat_tool_ind
	
	if g_model_type_ind == 0:
		input_dim = [g_input_3D_dim, g_input_3D_dim, g_input_3D_dim, g_input_feat_size[g_feat_tool_ind]]
	elif g_model_type_ind == 1:
		input_dim = [1350, g_input_feat_size[g_feat_tool_ind]]  # check csv to check # ligand atoms and pocket atoms
	return input_dim

def get_model_subdir():
	global g_model_type_ind
	global g_target_dataset
	global g_target_trainval_type
	global g_model_3dcnn_type_list
	global g_model_3dcnn_type_ind
	global g_model_snet_type_list
	global g_model_snet_type_ind
	global g_feat_suffix
	global g_3D_suffix
	
	now = datetime.datetime.now()
	date_str = "%04d%02d%02d" % (now.year, now.month, now.day)
	date_str = "20191009"
	#date_str = "20200101"
	if g_model_type_ind == 0:
		model_subdir = "%s_%s_%s_result_%s" % (g_feat_suffix, g_3D_suffix, g_model_3dcnn_type_list[g_model_3dcnn_type_ind], date_str)
	elif g_model_type_ind == 1:
		model_subdir = "%s_%s_result_%s" % (g_feat_suffix, g_model_snet_type_list[g_model_snet_type_ind], date_str)
	return model_subdir


def rotate_3D(input_data):
	rotation_angle = np.random.uniform() * 2 * np.pi
	cosval = np.cos(rotation_angle)
	sinval = np.sin(rotation_angle)
	rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
	#rotated_data = np.zeros(input_data.shape, dtype=np.float32)
	rotated_data = np.dot(input_data, rotation_matrix)
	return rotated_data


def get_3D_bound(xyz_array):
    xmin = min(xyz_array[:, 0])
    ymin = min(xyz_array[:, 1])
    zmin = min(xyz_array[:, 2])
    xmax = max(xyz_array[:, 0])
    ymax = max(xyz_array[:, 1])
    zmax = max(xyz_array[:, 2])
    return xmin, ymin, zmin, xmax, ymax, zmax


def get_3D_all2__(xyz, feat, vol_dim, xmin, ymin, zmin, xmax, ymax, zmax, atom_radii=None, atom_radius=1, sigma=0):

	# initialize volume
	vol_data = np.zeros((vol_dim[0], vol_dim[1], vol_dim[2], vol_dim[3]), dtype=np.float32)

	# voxel size (assum voxel size is the same in all axis
	vox_size = (zmax - zmin) / vol_dim[0]

	# assign each atom to voxels
	for ind in range(xyz.shape[0]):
		x = xyz[ind, 0]
		y = xyz[ind, 1]
		z = xyz[ind, 2]
		if x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax:
			continue

		# compute van der Waals radius and atomic density, use 1 if not available
		if not atom_radii is None:
			vdw_radius = atom_radii[ind]
			atom_radius = 1 + vdw_radius * vox_size

		cx = (x - xmin) / (xmax - xmin) * (vol_dim[2] - 1)
		cy = (y - ymin) / (ymax - ymin) * (vol_dim[1] - 1)
		cz = (z - zmin) / (zmax - zmin) * (vol_dim[0] - 1)

		vx_from = max(0, int(cx - atom_radius))
		vx_to = min(vol_dim[2] - 1, int(cx + atom_radius))
		vy_from = max(0, int(cy - atom_radius))
		vy_to = min(vol_dim[1] - 1, int(cy + atom_radius))
		vz_from = max(0, int(cz - atom_radius))
		vz_to = min(vol_dim[0] - 1, int(cz + atom_radius))

		for vz in range(vz_from, vz_to + 1):
			for vy in range(vy_from, vy_to + 1):
				for vx in range(vx_from, vx_to + 1):
						vol_data[vz, vy, vx, :] += feat[ind, :]

	# gaussian filter
	if sigma > 0:
		for i in range(vol_data.shape[-1]):
			vol_data[:,:,:,i] = sp.ndimage.filters.gaussian_filter(vol_data[:,:,:,i], sigma=sigma, truncate=2)

	return vol_data
	

def get_3D_all2(xyz, feat, vol_dim, relative_size=True, size_angstrom=48, atom_radii=None, atom_radius=1, sigma=0):

	# get 3d bounding box
	xmin, ymin, zmin, xmax, ymax, zmax = get_3D_bound(xyz)
	
	# initialize volume
	vol_data = np.zeros((vol_dim[0], vol_dim[1], vol_dim[2], vol_dim[3]), dtype=np.float32)

	if relative_size:
		# voxel size (assum voxel size is the same in all axis
		vox_size = float(zmax - zmin) / float(vol_dim[0])
	else:
		vox_size = float(size_angstrom) / float(vol_dim[0])
		xmid = (xmin + xmax) / 2.0
		ymid = (ymin + ymax) / 2.0
		zmid = (zmin + zmax) / 2.0
		xmin = xmid - (size_angstrom / 2)
		ymin = ymid - (size_angstrom / 2)
		zmin = zmid - (size_angstrom / 2)
		xmax = xmid + (size_angstrom / 2)
		ymax = ymid + (size_angstrom / 2)
		zmax = zmid + (size_angstrom / 2)
		vox_size2 = float(size_angstrom) / float(vol_dim[0])
		#print(vox_size, vox_size2)

	# assign each atom to voxels
	for ind in range(xyz.shape[0]):
		x = xyz[ind, 0]
		y = xyz[ind, 1]
		z = xyz[ind, 2]
		if x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax:
			continue

		# compute van der Waals radius and atomic density, use 1 if not available
		if not atom_radii is None:
			vdw_radius = atom_radii[ind]
			atom_radius = 1 + vdw_radius * vox_size

		cx = (x - xmin) / (xmax - xmin) * (vol_dim[2] - 1)
		cy = (y - ymin) / (ymax - ymin) * (vol_dim[1] - 1)
		cz = (z - zmin) / (zmax - zmin) * (vol_dim[0] - 1)

		vx_from = max(0, int(cx - atom_radius))
		vx_to = min(vol_dim[2] - 1, int(cx + atom_radius))
		vy_from = max(0, int(cy - atom_radius))
		vy_to = min(vol_dim[1] - 1, int(cy + atom_radius))
		vz_from = max(0, int(cz - atom_radius))
		vz_to = min(vol_dim[0] - 1, int(cz + atom_radius))

		for vz in range(vz_from, vz_to + 1):
			for vy in range(vy_from, vy_to + 1):
				for vx in range(vx_from, vx_to + 1):
						vol_data[vz, vy, vx, :] += feat[ind, :]

	# gaussian filter
	if sigma > 0:
		for i in range(vol_data.shape[-1]):
			vol_data[:,:,:,i] = sp.ndimage.filters.gaussian_filter(vol_data[:,:,:,i], sigma=sigma, truncate=2)

	return vol_data


def model_3dcnn(model_name, img_data, train_mode, reuse):
	with tf.variable_scope(model_name, reuse=reuse):
		x = img_data
		print(x.shape)
		
		conv1_w = weight_var_selu([7,7,7,g_input_dim[3],96],name="conv1_w")
		conv1_b = bias_var([96],name="conv1_b")
		conv1_z = conv3d(x, conv1_w, strides=(1,2,2,2,1)) + conv1_b
		conv1_h = bn(tf.nn.relu(conv1_z), train_mode,"conv1_bn")
		print(conv1_h.shape)

		conv2_w = weight_var_selu([7,7,7,96,128],name="conv2_w")
		conv2_b = bias_var([128],name="conv2_b")
		conv2_z = conv3d(conv1_h, conv2_w, strides=(1,3,3,3,1)) + conv2_b
		conv2_h = bn(tf.nn.relu(conv2_z), train_mode,"conv2_bn")
		print(conv2_h.shape)
		
		pool2_h = avg_pool_2x2x2(conv2_h)
		print(pool2_h.shape)
		
		conv3_w = weight_var_selu([5,5,5,128,128],name="conv3_w")
		conv3_b = bias_var([128],name="conv3_b")
		conv3_z = conv3d(pool2_h, conv3_w, strides=(1,2,2,2,1)) + conv3_b
		conv3_h = bn(tf.nn.relu(conv3_z), train_mode,"conv3_bn")
		print(conv3_h.shape)
		
		pool3_h = avg_pool_2x2x2(conv3_h)
		print(pool3_h.shape)
		
		pool3_h_dim = (pool3_h.shape[1] * pool3_h.shape[2] * pool3_h.shape[3] * pool3_h.shape[4])
		flatten_h = tf.reshape(pool3_h, [tf.shape(pool3_h)[0], int(pool3_h_dim)])
		print(flatten_h.shape)
		
		#fc1_w = weight_var([int(flatten_h.shape[1]), 2], stddev=0.01, name="fc1_w")
		#fc1_b = bias_var([2], name="fc1_b")
		#fc1_z = tf.matmul(flatten_h, fc1_w) + fc1_b
		#print(fc1_z.shape)
	#return fc1_z

		fc1_w = weight_var([int(flatten_h.shape[1]), 10], stddev=0.01, name="fc1_w")
		fc1_b = bias_var([10], name="fc1_b")
		fc1_z = tf.matmul(flatten_h, fc1_w) + fc1_b
		fc1_h = bn(tf.nn.relu(fc1_z), train_mode,"fc1_bn")
		print(fc1_h.shape)

		fc2_w = weight_var([10, 1], stddev=0.01, name="fc2_w")
		fc2_b = bias_var([1], name="fc2_b")
		fc2_z = tf.matmul(fc1_h, fc2_w) + fc2_b
		print(fc2_z.shape)
			
	return fc2_z


def model_3dcnn_res(model_name, img_data, train_mode, reuse):
	if g_feat_tool_ind == 0:
		num_filters = [64, 128, 256]
	else:
		num_filters = [96, 128, 128]

	with tf.variable_scope(model_name, reuse=reuse):
		x = img_data
		print(x.shape)
		
		conv1_w = weight_var_selu([7,7,7,g_input_dim[3],num_filters[0]],name="conv1_w")
		conv1_b = bias_var([num_filters[0]],name="conv1_b")
		conv1_z = conv3d(x, conv1_w, strides=(1,2,2,2,1)) + conv1_b
		conv1_h = bn(tf.nn.relu(conv1_z), train_mode,"conv1_bn")
		print(conv1_h.shape)
		
		conv1_res1_w = weight_var_selu([7,7,7,num_filters[0],num_filters[0]], name="conv1_res1_w")
		conv1_res1_b = bias_var([num_filters[0]], name="conv1_res1_b")
		conv1_res1_z = conv3d(conv1_h, conv1_res1_w, strides=(1,1,1,1,1)) + conv1_res1_b
		conv1_res1_h = bn(tf.nn.relu(conv1_res1_z), train_mode,"conv1_res1_bn")
		print(conv1_res1_h.shape)
		conv1_res1_h2 = conv1_res1_h + conv1_h
		print(conv1_res1_h2.shape)
		
		conv1_res2_w = weight_var_selu([7,7,7,num_filters[0],num_filters[0]], name="conv1_res2_w")
		conv1_res2_b = bias_var([num_filters[0]], name="conv1_res2_b")
		conv1_res2_z = conv3d(conv1_res1_h2, conv1_res2_w, strides=(1,1,1,1,1)) + conv1_res2_b
		conv1_res2_h = bn(tf.nn.relu(conv1_res2_z), train_mode,"conv1_res2_bn")
		print(conv1_res2_h.shape)
		conv1_res2_h2 = conv1_res2_h + conv1_h
		print(conv1_res2_h2.shape)
		
		conv2_w = weight_var_selu([7,7,7,num_filters[0],num_filters[1]],name="conv2_w")
		conv2_b = bias_var([num_filters[1]],name="conv2_b")
		conv2_z = conv3d(conv1_res2_h2, conv2_w, strides=(1,3,3,3,1)) + conv2_b
		conv2_h = bn(tf.nn.relu(conv2_z), train_mode, "conv2_bn")
		print(conv2_h.shape)
		
		pool2_h = avg_pool_2x2x2(conv2_h)
		print(pool2_h.shape)
		
		conv3_w = weight_var_selu([5,5,5,num_filters[1],num_filters[2]],name="conv3_w")
		conv3_b = bias_var([num_filters[2]],name="conv3_b")
		conv3_z = conv3d(pool2_h, conv3_w, strides=(1,2,2,2,1)) + conv3_b
		conv3_h = bn(tf.nn.relu(conv3_z), train_mode, "conv3_bn")
		print(conv3_h.shape)
		
		pool3_h = avg_pool_2x2x2(conv3_h)
		print(pool3_h.shape)
		
		pool3_h_dim = (pool3_h.shape[1] * pool3_h.shape[2] * pool3_h.shape[3] * pool3_h.shape[4])
		flatten_h = tf.reshape(pool3_h, [tf.shape(pool3_h)[0], int(pool3_h_dim)], name="flatten_h")
		print(flatten_h.shape)
		
		#fc1_w = weight_var([int(flatten_h.shape[1]), 2], stddev=0.01, name="fc1_w")
		#fc1_b = bias_var([2], name="fc1_b")
		#fc1_z = tf.matmul(flatten_h, fc1_w) + fc1_b
		#print(fc1_z.shape)
		#return fc1_z
		
		fc1_w = weight_var([int(flatten_h.shape[1]), 10], stddev=0.01, name="fc1_w")
		fc1_b = bias_var([10], name="fc1_b")
		fc1_z = tf.matmul(flatten_h, fc1_w) + fc1_b
		fc1_y = tf.nn.relu(fc1_z)
		fc1_h = bn(fc1_y, train_mode, "fc1_bn")
		print(fc1_h.shape)
		
		fc2_w = weight_var([10, 1], stddev=0.01, name="fc2_w")
		fc2_b = bias_var([1], name="fc2_b")
		fc2_z = tf.matmul(fc1_h, fc2_w) + fc2_b
		print(fc2_z.shape)
	
	return fc2_z, [fc1_h, fc1_y, fc1_z, flatten_h, pool3_h, conv3_h, pool2_h, conv2_h, conv1_res2_h2, conv1_res2_h, conv1_res1_h2, conv1_res1_h, conv1_h]


def model_3dcnn_res2(model_name, img_data, train_mode, reuse):
	with tf.variable_scope(model_name, reuse=reuse):
		x = img_data
		print(x.shape)
		
		conv1_w = weight_var_selu([7,7,7,g_input_dim[3],96],name="conv1_w")
		conv1_b = bias_var([96],name="conv1_b")
		conv1_z = conv3d(x, conv1_w, strides=(1,2,2,2,1)) + conv1_b
		conv1_h = bn(tf.nn.relu(conv1_z), train_mode,"conv1_bn")
		print(conv1_h.shape)
		
		conv1_res1_w = weight_var_selu([7,7,7,96,96], name="conv1_res1_w")
		conv1_res1_b = bias_var([96], name="conv1_res1_b")
		conv1_res1_z = conv3d(conv1_h, conv1_res1_w, strides=(1,1,1,1,1)) + conv1_res1_b
		conv1_res1_h = bn(tf.nn.relu(conv1_res1_z), train_mode,"conv1_res1_bn")
		print(conv1_res1_h.shape)
		conv1_res1_h2 = conv1_res1_h + conv1_h
		print(conv1_res1_h2.shape)
		
		conv1_res2_w = weight_var_selu([5,5,5,96,96], name="conv1_res2_w")
		conv1_res2_b = bias_var([96], name="conv1_res2_b")
		conv1_res2_z = conv3d(conv1_res1_h2, conv1_res2_w, strides=(1,1,1,1,1)) + conv1_res2_b
		conv1_res2_h = bn(tf.nn.relu(conv1_res2_z), train_mode,"conv1_res2_bn")
		print(conv1_res2_h.shape)
		conv1_res2_h2 = conv1_res2_h + conv1_h
		print(conv1_res2_h2.shape)

		conv1_res3_w = weight_var_selu([3,3,3,96,96], name="conv1_res3_w")
		conv1_res3_b = bias_var([96], name="conv1_res3_b")
		conv1_res3_z = conv3d(conv1_res2_h2, conv1_res3_w, strides=(1,1,1,1,1)) + conv1_res3_b
		conv1_res3_h = bn(tf.nn.relu(conv1_res3_z), train_mode,"conv1_res3_bn")
		print(conv1_res3_h.shape)
		conv1_res3_h2 = conv1_res3_h + conv1_h
		print(conv1_res3_h2.shape)

		conv2_w = weight_var_selu([5,5,5,96,96],name="conv2_w")
		conv2_b = bias_var([96],name="conv2_b")
		conv2_z = conv3d(conv1_res3_h2, conv2_w, strides=(1,3,3,3,1)) + conv2_b
		conv2_h = bn(tf.nn.relu(conv2_z), train_mode, "conv2_bn")
		print(conv2_h.shape)
		
		pool2_h = max_pool_2x2x2(conv2_h)
		print(pool2_h.shape)
		
		conv3_w = weight_var_selu([3,3,3,96,128],name="conv3_w")
		conv3_b = bias_var([128],name="conv3_b")
		conv3_z = conv3d(pool2_h, conv3_w, strides=(1,2,2,2,1)) + conv3_b
		conv3_h = bn(tf.nn.relu(conv3_z), train_mode, "conv3_bn")
		print(conv3_h.shape)
		
		pool3_h = max_pool_2x2x2(conv3_h)
		print(pool3_h.shape)
		
		pool3_h_dim = (pool3_h.shape[1] * pool3_h.shape[2] * pool3_h.shape[3] * pool3_h.shape[4])
		flatten_h = tf.reshape(pool3_h, [tf.shape(pool3_h)[0], int(pool3_h_dim)], name="flatten_h")
		print(flatten_h.shape)
		
		#fc1_w = weight_var([int(flatten_h.shape[1]), 2], stddev=0.01, name="fc1_w")
		#fc1_b = bias_var([2], name="fc1_b")
		#fc1_z = tf.matmul(flatten_h, fc1_w) + fc1_b
		#print(fc1_z.shape)
		#return fc1_z
		
		fc1_w = weight_var([int(flatten_h.shape[1]), 10], stddev=0.01, name="fc1_w")
		fc1_b = bias_var([10], name="fc1_b")
		fc1_z = tf.matmul(flatten_h, fc1_w) + fc1_b
		fc1_h = bn(tf.nn.relu(fc1_z), train_mode, "fc1_bn")
		print(fc1_h.shape)
		
		fc2_w = weight_var([10, 1], stddev=0.01, name="fc2_w")
		fc2_b = bias_var([1], name="fc2_b")
		fc2_z = tf.matmul(fc1_h, fc2_w) + fc2_b
		print(fc2_z.shape)
	
	return fc2_z, [fc1_h, fc1_z, flatten_h, pool3_h, conv3_h, pool2_h, conv2_h, conv1_res2_h2, conv1_res2_h, conv1_res1_h2, conv1_res1_h, conv1_h]


def model_3dcnn_res3(model_name, img_data, train_mode, reuse):
	with tf.variable_scope(model_name, reuse=reuse):
		x = img_data
		print(x.shape)
		
		conv1_w = weight_var_selu([7,7,7,g_input_dim[3],96],name="conv1_w")
		conv1_b = bias_var([96],name="conv1_b")
		conv1_z = conv3d(x, conv1_w, strides=(1,2,2,2,1)) + conv1_b
		conv1_h = bn(tf.nn.relu(conv1_z), train_mode,"conv1_bn")
		print(conv1_h.shape)

		conv1_res1_w = weight_var_selu([7,7,7,96,96], name="conv1_res1_w")
		conv1_res1_b = bias_var([96], name="conv1_res1_b")
		conv1_res1_z = conv3d(conv1_h, conv1_res1_w, strides=(1,1,1,1,1)) + conv1_res1_b
		conv1_res1_h = bn(tf.nn.relu(conv1_res1_z), train_mode,"conv1_res1_bn")
		print(conv1_res1_h.shape)
		conv1_res1_h2 = conv1_res1_h + conv1_h
		print(conv1_res1_h2.shape)
		
		conv1_res2_w = weight_var_selu([5,5,5,96,96], name="conv1_res2_w")
		conv1_res2_b = bias_var([96], name="conv1_res2_b")
		conv1_res2_z = conv3d(conv1_res1_h2, conv1_res2_w, strides=(1,1,1,1,1)) + conv1_res2_b
		conv1_res2_h = bn(tf.nn.relu(conv1_res2_z), train_mode,"conv1_res2_bn")
		print(conv1_res2_h.shape)
		conv1_res2_h2 = conv1_res2_h + conv1_h
		print(conv1_res2_h2.shape)

		pool1_h = max_pool_2x2x2(conv1_res2_h2)
		print(pool1_h.shape)

		conv2_w = weight_var_selu([5,5,5,96,96],name="conv2_w")
		conv2_b = bias_var([96],name="conv2_b")
		conv2_z = conv3d(pool1_h, conv2_w, strides=(1,2,2,2,1)) + conv2_b
		conv2_h = bn(tf.nn.relu(conv2_z), train_mode, "conv2_bn")
		print(conv2_h.shape)

		conv2_res1_w = weight_var_selu([5,5,5,96,96], name="conv2_res1_w")
		conv2_res1_b = bias_var([96], name="conv2_res1_b")
		conv2_res1_z = conv3d(conv2_h, conv2_res1_w, strides=(1,1,1,1,1)) + conv2_res1_b
		conv2_res1_h = bn(tf.nn.relu(conv2_res1_z), train_mode,"conv2_res1_bn")
		print(conv2_res1_h.shape)
		conv2_res1_h2 = conv2_res1_h + conv2_h
		print(conv2_res1_h2.shape)
		
		conv2_res2_w = weight_var_selu([3,3,3,96,96], name="conv2_res2_w")
		conv2_res2_b = bias_var([96], name="conv2_res2_b")
		conv2_res2_z = conv3d(conv2_res1_h2, conv2_res2_w, strides=(1,1,1,1,1)) + conv2_res2_b
		conv2_res2_h = bn(tf.nn.relu(conv2_res2_z), train_mode,"conv2_res2_bn")
		print(conv2_res2_h.shape)
		conv2_res2_h2 = conv2_res2_h + conv2_h
		print(conv2_res2_h2.shape)

		pool2_h = max_pool_2x2x2(conv2_res2_h2)
		print(pool2_h.shape)

		conv3_w = weight_var_selu([3,3,3,96,128],name="conv3_w")
		conv3_b = bias_var([128],name="conv3_b")
		conv3_z = conv3d(pool2_h, conv3_w, strides=(1,1,1,1,1)) + conv3_b
		conv3_h = bn(tf.nn.relu(conv3_z), train_mode, "conv3_bn")
		print(conv3_h.shape)

		pool3_h = max_pool_2x2x2(conv3_h)
		print(pool3_h.shape)

		pool3_h_dim = (pool3_h.shape[1] * pool3_h.shape[2] * pool3_h.shape[3] * pool3_h.shape[4])
		flatten_h = tf.reshape(pool3_h, [tf.shape(pool3_h)[0], int(pool3_h_dim)], name="flatten_h")
		print(flatten_h.shape)
		
		fc1_w = weight_var([int(flatten_h.shape[1]), 10], stddev=0.01, name="fc1_w")
		fc1_b = bias_var([10], name="fc1_b")
		fc1_z = tf.matmul(flatten_h, fc1_w) + fc1_b
		fc1_h = bn(tf.nn.relu(fc1_z), train_mode, "fc1_bn")
		print(fc1_h.shape)
		
		fc2_w = weight_var([10, 1], stddev=0.01, name="fc2_w")
		fc2_b = bias_var([1], name="fc2_b")
		fc2_z = tf.matmul(fc1_h, fc2_w) + fc2_b
		print(fc2_z.shape)
	
	return fc2_z, [fc1_h, fc1_z, flatten_h, pool3_h, conv3_h, pool2_h, conv2_h, conv1_res2_h2, conv1_res2_h, conv1_res1_h2, conv1_res1_h, conv1_h]


def model_3dcnn_res4(model_name, img_data, train_mode, reuse):
	if g_feat_tool_ind == 0:
		num_filters = [96, 128, 256, 512]
	else:
		num_filters = [96, 128, 128]
	dropout = 0.5

	with tf.variable_scope(model_name, reuse=reuse):
		x = img_data
		print(x.shape)
		
		conv1_w = weight_var_selu([9,9,9,g_input_dim[3],num_filters[0]],name="conv1_w")
		conv1_b = bias_var([num_filters[0]],name="conv1_b")
		conv1_z = conv3d(x, conv1_w, strides=(1,3,3,3,1)) + conv1_b
		conv1_h = bn(tf.nn.relu(conv1_z), train_mode,"conv1_bn")
		print(conv1_h.shape)
		
		conv1_res1_w = weight_var_selu([9,9,9,num_filters[0],num_filters[0]], name="conv1_res1_w")
		conv1_res1_b = bias_var([num_filters[0]], name="conv1_res1_b")
		conv1_res1_z = conv3d(conv1_h, conv1_res1_w, strides=(1,1,1,1,1)) + conv1_res1_b
		conv1_res1_h = bn(tf.nn.relu(conv1_res1_z), train_mode,"conv1_res1_bn")
		print(conv1_res1_h.shape)
		conv1_res1_h2 = conv1_res1_h + conv1_h
		print(conv1_res1_h2.shape)
		
		conv1_res2_w = weight_var_selu([9,9,9,num_filters[0],num_filters[0]], name="conv1_res2_w")
		conv1_res2_b = bias_var([num_filters[0]], name="conv1_res2_b")
		conv1_res2_z = conv3d(conv1_res1_h2, conv1_res2_w, strides=(1,1,1,1,1)) + conv1_res2_b
		conv1_res2_h = bn(tf.nn.relu(conv1_res2_z), train_mode,"conv1_res2_bn")
		print(conv1_res2_h.shape)
		conv1_res2_h2 = conv1_res2_h + conv1_res1_h + conv1_h
		print(conv1_res2_h2.shape)
		

		conv2_w = weight_var_selu([7,7,7,num_filters[0],num_filters[1]],name="conv2_w")
		conv2_b = bias_var([num_filters[1]],name="conv2_b")
		conv2_z = conv3d(conv1_res2_h2, conv2_w, strides=(1,3,3,3,1)) + conv2_b
		conv2_h = bn(tf.nn.relu(conv2_z), train_mode, "conv2_bn")
		print(conv2_h.shape)

		conv2_res1_w = weight_var_selu([7,7,7,num_filters[1],num_filters[1]], name="conv2_res1_w")
		conv2_res1_b = bias_var([num_filters[1]], name="conv2_res1_b")
		conv2_res1_z = conv3d(conv2_h, conv2_res1_w, strides=(1,1,1,1,1)) + conv2_res1_b
		conv2_res1_h = bn(tf.nn.relu(conv2_res1_z), train_mode,"conv2_res1_bn")
		print(conv2_res1_h.shape)
		conv2_res1_h2 = conv2_res1_h + conv2_h
		print(conv2_res1_h2.shape)

		conv2_res2_w = weight_var_selu([7,7,7,num_filters[1],num_filters[1]], name="conv2_res2_w")
		conv2_res2_b = bias_var([num_filters[1]], name="conv2_res2_b")
		conv2_res2_z = conv3d(conv2_res1_h2, conv2_res2_w, strides=(1,1,1,1,1)) + conv2_res2_b
		conv2_res2_h = bn(tf.nn.relu(conv2_res2_z), train_mode,"conv2_res2_bn")
		print(conv2_res2_h.shape)
		conv2_res2_h2 = conv2_res2_h + conv2_res1_h + conv2_h
		print(conv2_res2_h2.shape)
		
	
		conv3_w = weight_var_selu([5,5,5,num_filters[1],num_filters[2]],name="conv3_w")
		conv3_b = bias_var([num_filters[2]],name="conv3_b")
		conv3_z = conv3d(conv2_res2_h2, conv3_w, strides=(1,2,2,2,1)) + conv3_b
		conv3_h = bn(tf.nn.dropout(tf.nn.relu(conv3_z), keep_prob=dropout), train_mode, "conv3_bn")
		print(conv3_h.shape)
	
	
		conv4_w = weight_var_selu([3,3,3,num_filters[2],num_filters[3]],name="conv4_w")
		conv4_b = bias_var([num_filters[3]],name="conv4_b")
		conv4_z = conv3d(conv3_h, conv4_w, strides=(1,1,1,1,1)) + conv4_b
		conv4_h = bn(tf.nn.relu(conv4_z), train_mode, "conv4_bn")
		print(conv4_h.shape)

		pool4_h = max_pool_2x2x2(conv4_h)
		print(pool4_h.shape)

		pool4_h_dim = (pool4_h.shape[1] * pool4_h.shape[2] * pool4_h.shape[3] * pool4_h.shape[4])
		flatten_h = tf.reshape(pool4_h, [tf.shape(pool4_h)[0], int(pool4_h_dim)], name="flatten_h")
		print(flatten_h.shape)
		
		#fc1_w = weight_var([int(flatten_h.shape[1]), 2], stddev=0.01, name="fc1_w")
		#fc1_b = bias_var([2], name="fc1_b")
		#fc1_z = tf.matmul(flatten_h, fc1_w) + fc1_b
		#print(fc1_z.shape)
		#return fc1_z
		
		fc1_w = weight_var([int(flatten_h.shape[1]), 50], stddev=0.01, name="fc1_w")
		fc1_b = bias_var([50], name="fc1_b")
		fc1_z = tf.matmul(flatten_h, fc1_w) + fc1_b
		fc1_y = tf.nn.dropout(tf.nn.relu(fc1_z), keep_prob=dropout)
		fc1_h = bn(fc1_y, train_mode, "fc1_bn")
		print(fc1_h.shape)

		
		fc2_w = weight_var([50, 1], stddev=0.01, name="fc2_w")
		fc2_b = bias_var([1], name="fc2_b")
		fc2_z = tf.matmul(fc1_h, fc2_w) + fc2_b
		print(fc2_z.shape)
	
	return fc2_z, [fc1_h, fc1_y, fc1_z, flatten_h]


def model_3dcnn_atomnet(model_name, img_data, train_mode, reuse):
	with tf.variable_scope(model_name, reuse=reuse):
		keep_prob = 0.3
	
		x = img_data
		print(x.shape)
		
		conv1_w = weight_var_selu([5,5,5,g_input_dim[3],20],name="conv1_w")
		conv1_b = bias_var([20],name="conv1_b")
		conv1_z = conv3d(x, conv1_w, strides=(1,1,1,1,1)) + conv1_b
		conv1_h = tf.nn.relu(conv1_z)
		print(conv1_h.shape)
		
		pool1_h = avg_pool_2x2x2(conv1_h)
		print(pool1_h.shape)
		
		conv2_w = weight_var_selu([3,3,3,20,30],name="conv2_w")
		conv2_b = bias_var([30],name="conv2_b")
		conv2_z = conv3d(pool1_h, conv2_w, strides=(1,1,1,1,1)) + conv2_b
		conv2_h = tf.nn.relu(conv2_z)
		print(conv2_h.shape)
		
		pool2_h = avg_pool_2x2x2(conv2_h)
		print(pool2_h.shape)
		
		conv3_w = weight_var_selu([2,2,2,30,40],name="conv3_w")
		conv3_b = bias_var([40],name="conv3_b")
		conv3_z = conv3d(pool2_h, conv3_w, strides=(1,1,1,1,1)) + conv3_b
		conv3_h = tf.nn.relu(conv3_z)
		print(conv3_h.shape)
		
		pool3_h = avg_pool_2x2x2(conv3_h)
		print(pool3_h.shape)
		
		conv4_w = weight_var_selu([2,2,2,40,50],name="conv4_w")
		conv4_b = bias_var([50],name="conv4_b")
		conv4_z = conv3d(pool3_h, conv4_w, strides=(1,1,1,1,1)) + conv4_b
		conv4_h = tf.nn.relu(conv4_z)
		print(conv4_h.shape)
		
		#pool4_h = avg_pool_2x2x2(conv4_h)
		#print(pool4_h.shape)
		pool4_h = conv4_h
		
		conv5_w = weight_var_selu([2,2,2,50,60],name="conv5_w")
		conv5_b = bias_var([60],name="conv5_b")
		conv5_z = conv3d(pool4_h, conv5_w, strides=(1,1,1,1,1)) + conv5_b
		conv5_h = tf.nn.relu(conv5_z)
		print(conv5_h.shape)
		
		#pool5_h = avg_pool_2x2x2(conv5_h)
		#print(pool5_h.shape)
		pool5_h = conv5_h
		
		pool5_h_dim = (pool5_h.shape[1] * pool5_h.shape[2] * pool5_h.shape[3] * pool5_h.shape[4])
		flatten_h = tf.reshape(pool5_h, [tf.shape(pool5_h)[0], int(pool5_h_dim)], name="flatten_h")
		print(flatten_h.shape)
		
		fc1_w = weight_var([int(flatten_h.shape[1]), 1024], stddev=0.01, name="fc1_w")
		fc1_b = bias_var([1024], name="fc1_b")
		fc1_z = tf.matmul(flatten_h, fc1_w) + fc1_b
		fc1_h = tf.nn.dropout(tf.nn.relu(fc1_z), keep_prob)
		print(fc1_h.shape)

		fc2_w = weight_var([1024, 256], stddev=0.01, name="fc2_w")
		fc2_b = bias_var([256], name="fc2_b")
		fc2_z = tf.matmul(fc1_h, fc2_w) + fc2_b
		fc2_h = tf.nn.relu(fc2_z)
		print(fc2_h.shape)

		fc3_w = weight_var([256, 1], stddev=0.01, name="fc3_w")
		fc3_b = bias_var([1], name="fc3_b")
		fc3_z = tf.matmul(fc2_h, fc3_w) + fc3_b
		print(fc3_z.shape)
	
	return fc3_z, [fc2_h, fc2_z, fc1_h, fc1_z, flatten_h]


def model_snet(model_name, input_data, train_mode, reuse):
	input_data = tf.expand_dims(input_data, -1)

	num_filters = 1024
	filter_size = 1
	
	sequence_length = g_input_dim[0]
	embedding_size = g_input_feat_size[g_feat_tool_ind]
	pooled_outputs = []

	with tf.variable_scope(model_name, reuse=reuse):
		W1 = tf.Variable(tf.truncated_normal((filter_size, embedding_size, 1, num_filters), stddev=0.1), name="W_conv1")
		b1 = bias_var([num_filters],name="b_conv1")
		conv1 = tf.nn.conv2d(input_data, W1, strides=[1, 1, 1, 1], padding="VALID", name="conv1")
		h1 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(conv1, b1), name="relu"),keep_prob=1)
		print(h1.shape)

		W2 = tf.Variable(tf.truncated_normal((1, 1, num_filters, num_filters), stddev=0.1), name="W_conv2")
		b2 = bias_var([num_filters],name="b_conv2")
		conv2 = tf.nn.conv2d(h1, W2, strides=[1, 1, 1, 1], padding="VALID", name="conv2")
		h2 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(conv2, b2), name="relu"),keep_prob=1)
		print(h2.shape)
		h2_res = h1 + h2
		
		W3 = tf.Variable(tf.truncated_normal((1, 1, num_filters, num_filters), stddev=0.1), name="W_conv3")
		b3 = bias_var([num_filters],name="b_conv3")
		conv3 = tf.nn.conv2d(h2_res, W3, strides=[1, 1, 1, 1], padding="VALID", name="conv3")
		h3 = tf.nn.dropout(tf.nn.relu(tf.nn.bias_add(conv3, b3), name="relu"),keep_prob=1)
		print(h3.shape)
		h3_res = h2_res + h3

		pooled = tf.nn.max_pool(h3_res, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
		print(pooled.shape)
	
		pooled_flat = tf.reshape(pooled, [-1, num_filters])
		print(pooled_flat.shape)

		fc1_w = weight_var([int(pooled_flat.shape[1]), 100], stddev=0.01, name="fc1_w")
		fc1_b = bias_var([100], name="fc1_b")
		fc1_z = tf.matmul(pooled_flat, fc1_w) + fc1_b
		fc1_h = bn(tf.nn.relu(fc1_z), train_mode, "fc1_bn")
		print(fc1_h.shape)

		fc2_w = weight_var([100, 10], stddev=0.01, name="fc2_w")
		fc2_b = bias_var([10], name="fc2_b")
		fc2_z = tf.matmul(fc1_h, fc2_w) + fc2_b
		fc2_h = bn(tf.nn.relu(fc2_z), train_mode, "fc2_bn")
		print(fc2_h.shape)
		
		fc3_w = weight_var([10, 1], stddev=0.01, name="fc3_w")
		fc3_b = bias_var([1], name="fc3_b")
		fc3_z = tf.matmul(fc2_h, fc3_w) + fc3_b
		print(fc3_z.shape)
	
	return fc3_z, [fc2_h, fc1_h, pooled_flat]


def run_custom_batch(dnn, sess, saver, batch_ind, batch_size, x_batch, y_batch, custom_vars):
	
	print("[%d] extracting features... " % (batch_ind))
	pred, feat1, feat2 = sess.run([dnn.logit_ph, dnn.user_phs[1], dnn.user_phs[3]], feed_dict={dnn.input_ph: x_batch, dnn.training_phase_ph : 0})
	
	pred_arr = custom_vars[0]
	feat_arr1 = custom_vars[1]
	feat_arr2 = custom_vars[2]

	ind1 = (batch_ind) * batch_size
	ind2 = (batch_ind+1) * batch_size
	pred_arr[ind1:ind2] = pred
	feat_arr1[ind1:ind2,:] = feat1
	feat_arr2[ind1:ind2,:] = feat2


def input_reader(data_ind, main_dir, input_info):
	global g_input_hds
	
	if g_external_hd_type == 1:
		split = int(input_info.split('/')[0])
		lig_id = input_info.split('/')[2]
		if g_model_type_ind == 0 and g_input_3D_rotate == False:
			input_data = g_input_hds[split][lig_id]
		else:
			tool_str = g_feat_tool_list[g_feat_tool_ind]
			type_str = g_feat_type_list[g_feat_type_ind]
			input_data_ = g_input_hds[split][lig_id][tool_str][type_str]
			input_xyz = input_data_[:,0:3]
			input_feat = input_data_[:,3:]
			if g_model_type_ind == 0:
				if g_run_mode == 1:
					input_xyz = rotate_3D(input_xyz)
				xmin, ymin, zmin, xmax, ymax, zmax = get_3D_bound(input_xyz)
				input_data = get_3D_all2(input_xyz, input_feat, g_input_dim, xmin, ymin, zmin, xmax, ymax, zmax, g_input_3D_atom_radii, g_input_3D_atom_radius, g_input_3D_sigma)
			elif g_model_type_ind == 1:
				input_data = np.zeros((g_input_dim[0], g_input_dim[1]), dtype=np.float32)
				input_data[:input_feat.shape[0],:] = input_feat

	elif g_external_hd_type == 2:
		split = int(input_info.split('/')[0])
		lig_id_pose = input_info.split('/')[2]
		lig_id = '_'.join(lig_id_pose.split('_')[:-1])
		pose_id = lig_id_pose.split('_')[-1]
		input_hd = g_input_hds[split]
		input_data_ = input_hd[lig_id]["pybel"]["processed"]["docking"][pose_id]["data"]
		
		input_radii = None
		if g_input_3D_atom_radii:
			input_radii = input_data_.attrs['van_der_waals']
		input_xyz = input_data_[:,0:3]
		input_feat = input_data_[:,3:]
		xmin, ymin, zmin, xmax, ymax, zmax = get_3D_bound(input_xyz)
		#input_data = get_3D_all2(input_xyz, input_feat, g_input_dim, xmin, ymin, zmin, xmax, ymax, zmax, g_input_3D_atom_radii, g_input_3D_atom_radius, g_input_3D_sigma)
		input_data = get_3D_all2(input_xyz, input_feat, g_input_dim, g_input_3D_relative_size, g_input_3D_size_angstrom, \
						input_radii, g_input_3D_atom_radius, g_input_3D_sigma)

	elif g_external_hd_type == 3:
		split = int(input_info.split('/')[0])
		lig_id = input_info.split('/')[2]
		input_hd = g_input_hds[split]
		input_data_ = input_hd[lig_id]["pybel"]["processed"]["pdbbind"]["data"]
		
		input_radii = None
		if g_input_3D_atom_radii:
			input_radii = input_data_.attrs['van_der_waals']
		input_xyz = input_data_[:,0:3]
		input_feat = input_data_[:,3:]
		xmin, ymin, zmin, xmax, ymax, zmax = get_3D_bound(input_xyz)
		#input_data = get_3D_all2(input_xyz, input_feat, g_input_dim, xmin, ymin, zmin, xmax, ymax, zmax, g_input_3D_atom_radii, g_input_3D_atom_radius, g_input_3D_sigma)
		input_data = get_3D_all2(input_xyz, input_feat, g_input_dim, g_input_3D_relative_size, g_input_3D_size_angstrom, \
						input_radii, g_input_3D_atom_radius, g_input_3D_sigma)

	return input_data


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--main-dir", default=[], nargs="+", help="main dataset directory")
	parser.add_argument("--model-subdir", default=[], nargs="+", help="subdirectory storing models/results (under main_dir)")
	parser.add_argument("--run-mode", default=[], nargs="+", help="1: training, 2: test, 3: save features/sinogram")
	parser.add_argument("--external-dir", default=[], nargs="+")
	parser.add_argument("--external-hdftype", default=[], nargs="+")
	parser.add_argument("--external-csvfile", default=[], nargs="+")
	parser.add_argument("--external-trainhdf", default=[], nargs="+")
	parser.add_argument("--external-valhdf", default=[], nargs="+")
	parser.add_argument("--external-testhdf", default=[], nargs="+")
	parser.add_argument("--external-featprefix", default=[], nargs="+")
	args = parser.parse_args()

	global g_feat_suffix
	global g_3D_suffix
	global g_input_dim
	g_feat_suffix = get_feat_suffix()
	g_3D_suffix = get_3D_suffix()
	g_input_dim = get_input_dim()

	global g_main_dir
	global g_model_subdir
	global g_run_mode
	global g_external_dir
	global g_external_hd_type
	global g_external_hd_files
	global g_external_csv_file
	if args.main_dir:
		g_main_dir = args.main_dir[0]
	if args.model_subdir:
		g_model_subdir = args.model_subdir[0]
	else:
		g_model_subdir = get_model_subdir()
	if args.run_mode:
		g_run_mode = int(args.run_mode[0])
	if args.external_dir:
		g_external_dir = args.external_dir[0]
	if args.external_hdftype:
		g_external_hd_type = int(args.external_hdftype[0])
	if args.external_testhdf:
		g_external_hd_files = ['', '', args.external_testhdf[0]]
	if args.external_csvfile:
		g_external_csv_file = args.external_csvfile[0]
	if g_run_mode == 5 and args.external_featprefix:
		external_feat_prefix = args.external_featprefix[0]
	else:
		external_feat_prefix = g_model_subdir


	# load dataset
	global g_input_hds
	if g_run_mode == 4 or g_run_mode == 5:
		for ds_ind in range(3):
			if g_external_hd_files[ds_ind] != "":
				g_input_hds[ds_ind] = h5py.File(os.path.join(g_external_dir, g_external_hd_files[ds_ind]), 'r')
		if g_external_hd_type == 2 or g_external_hd_type == 3:
			g_external_csv_file = external_feat_prefix + ".csv"
			with open(os.path.join(g_external_dir, g_external_csv_file), 'w') as csv_fp:
				csv_writer = csv.writer(csv_fp, delimiter=',')
				csv_writer.writerow(g_csv_header)
			
				for ds_ind, input_hd in enumerate(g_input_hds):
					if input_hd == None:
						continue
					for lig_id in input_hd.keys():
						if g_external_hd_type == 2:
							pose_data = input_hd[lig_id]["pybel"]["processed"]["docking"]
							for pose_id in range(1,11):
								if not str(pose_id) in pose_data:
									continue
				
								lig_id_pose = lig_id + '_' + str(pose_id)
								lig_prefix = "%d/%s/%s" % (ds_ind, g_external_hd_files[ds_ind], lig_id_pose)
								lig_affinity = input_hd[lig_id].attrs["affinity"]
								csv_writer.writerow([lig_id_pose, lig_prefix, lig_affinity, ds_ind])
						else:
							lig_prefix = "%d/%s/%s" % (ds_ind, g_external_hd_files[ds_ind], lig_id)
							lig_affinity = input_hd[lig_id].attrs["affinity"]
							csv_writer.writerow([lig_id, lig_prefix, lig_affinity, ds_ind])
							
		
		data_reader = DataReader(g_external_dir, g_external_csv_file, g_csv_ind_input, g_csv_ind_output, g_csv_ind_split, g_input_dim, g_input_type, input_reader, g_output_dim, g_output_type, None)
			
	else:
		# load hd5
		if g_model_type_ind == 0 and g_input_3D_rotate == False:
			input_train_hd_file = "%s_%s_%s" % (g_feat_suffix, g_3D_suffix, g_input_train_hd_suffix)
			input_val_hd_file = "%s_%s_%s" % (g_feat_suffix, g_3D_suffix, g_input_val_hd_suffix)
			input_test_hd_file = "%s_%s_%s" % (g_feat_suffix, g_3D_suffix, g_input_test_hd_suffix)
			g_input_hds[0] = h5py.File(os.path.join(g_main_dir, input_train_hd_file), 'r')
			g_input_hds[1] = h5py.File(os.path.join(g_main_dir, input_val_hd_file), 'r')
			g_input_hds[2] = h5py.File(os.path.join(g_main_dir, input_test_hd_file), 'r')
		else:
			input_feat_train_hd_file = "%s_%s_%s" % (g_target_dataset, g_target_trainval_type, g_input_train_hd_suffix)
			input_feat_val_hd_file = "%s_%s_%s" % (g_target_dataset, g_target_trainval_type, g_input_val_hd_suffix)
			input_feat_test_hd_file = "%s_%s_core_%s" % (g_target_dataset, g_target_trainval_type, g_input_test_hd_suffix)
			g_input_hds[0] = h5py.File(os.path.join(g_main_dir, g_input_train_feat_hd_file), 'r')
			g_input_hds[1] = h5py.File(os.path.join(g_main_dir, g_input_train_feat_hd_file), 'r')
			g_input_hds[2] = h5py.File(os.path.join(g_main_dir, g_input_test_feat_hd_file), 'r')

		csv_file = "%s_%s_%s" % (g_feat_suffix, g_3D_suffix, g_csv_suffix)
		data_reader = DataReader(g_main_dir, csv_file, g_csv_ind_input, g_csv_ind_output, g_csv_ind_split, g_input_dim, g_input_type, input_reader, g_output_dim, g_output_type, None)

	
	# initialize CNN
	if g_model_type_ind == 0:
		if g_model_3dcnn_type_ind == 0:
			model = model_3dcnn
		elif  g_model_3dcnn_type_ind == 1:
			model = model_3dcnn_res
		elif  g_model_3dcnn_type_ind == 2:
			model = model_3dcnn_res2
		elif  g_model_3dcnn_type_ind == 3:
			model = model_3dcnn_res3
		elif  g_model_3dcnn_type_ind == 4:
			model = model_3dcnn_res4
		elif  g_model_3dcnn_type_ind == 5:
			model = model_3dcnn_atomnet
		model_name = g_model_3dcnn_type_list[g_model_3dcnn_type_ind]
	elif g_model_type_ind == 1:
		model = model_snet
		model_name = g_model_snet_type_list[g_model_snet_type_ind]

	cnn = DNN_General(data_reader, model, model_name, output_dir=os.path.join(g_main_dir, g_model_subdir), optimizer_info=g_optimizer_info, decay_info=g_decay_info, model_loss_info=g_loss_info)
	
	# train CNN
	if g_run_mode == 1:
		cnn.train(g_epoch_count, g_batch_size, g_online_batch_size, g_save_rate, g_verbose, g_val_each_epoch, 1)
	
	elif g_run_mode == 2:
		cnn.test(1, g_verbose, g_test_save_output)
	
	elif g_run_mode == 3 or g_run_mode == 5:
		if g_run_mode == 3:
			featdir = os.path.join(g_main_dir, g_model_subdir)
		else:
			featdir = g_external_dir

		if len(data_reader.train_list) > 0:
			print("extracting training sample features...")
			data_reader.begin_train(1, shuffle=False)
			train_pred_arr = np.ndarray(shape=(data_reader.train_batch_count), dtype=np.float32)
			train_feat_arr1 = np.ndarray(shape=(data_reader.train_batch_count, 10), dtype=np.float32)
			train_feat_arr2 = np.ndarray(shape=(data_reader.train_batch_count, 256), dtype=np.float32)
			custom_vars = [train_pred_arr, train_feat_arr1, train_feat_arr2]
			cnn.train_custom(1, False, run_custom_batch, custom_vars)
			np.save(os.path.join(featdir, '%s_train_pred.npy' % external_feat_prefix), train_pred_arr)
			np.save(os.path.join(featdir, '%s_train_fc10.npy' % external_feat_prefix), train_feat_arr1)
			np.save(os.path.join(featdir, '%s_train_fc256.npy' % external_feat_prefix), train_feat_arr2)
			print("training features saved")

		if len(data_reader.val_list) > 0:
			print("extracting validation sample features...")
			data_reader.begin_val(1)
			val_pred_arr = np.ndarray(shape=(data_reader.val_batch_count), dtype=np.float32)
			val_feat_arr1 = np.ndarray(shape=(data_reader.val_batch_count, 10), dtype=np.float32)
			val_feat_arr2 = np.ndarray(shape=(data_reader.val_batch_count, 256), dtype=np.float32)
			custom_vars = [val_pred_arr, val_feat_arr1, val_feat_arr2]
			cnn.val_custom(1, run_custom_batch, custom_vars)
			np.save(os.path.join(featdir, '%s_val_pred.npy' % external_feat_prefix), val_pred_arr)
			np.save(os.path.join(featdir, '%s_val_fc10.npy' % external_feat_prefix), val_feat_arr1)
			np.save(os.path.join(featdir, '%s_val_fc256.npy' % external_feat_prefix), val_feat_arr2)
			print("validation features saved")

		if len(data_reader.test_list) > 0:
			print("extracting test sample features...")
			data_reader.begin_test(1)
			test_pred_arr = np.ndarray(shape=(data_reader.test_batch_count), dtype=np.float32)
			test_feat_arr1 = np.ndarray(shape=(data_reader.test_batch_count, 10), dtype=np.float32)
			test_feat_arr2 = np.ndarray(shape=(data_reader.test_batch_count, 256), dtype=np.float32)
			custom_vars = [test_pred_arr, test_feat_arr1, test_feat_arr2]
			cnn.test_custom(1, run_custom_batch, custom_vars)
			if len(data_reader.train_list) > 0 or len(data_reader.val_list) > 0:
				np.save(os.path.join(featdir, '%s_test_pred.npy' % external_feat_prefix), test_pred_arr)
				np.save(os.path.join(featdir, '%s_test_fc10.npy' % external_feat_prefix), test_feat_arr1)
				np.save(os.path.join(featdir, '%s_test_fc256.npy' % external_feat_prefix), test_feat_arr2)
			else:
				np.save(os.path.join(featdir, '%s_pred.npy' % external_feat_prefix), test_pred_arr)
				np.save(os.path.join(featdir, '%s_fc10.npy' % external_feat_prefix), test_feat_arr1)
				np.save(os.path.join(featdir, '%s_fc256.npy' % external_feat_prefix), test_feat_arr2)
				with open(os.path.join(featdir, '%s_log' % external_feat_prefix), 'w') as f:
					f.write("done\n")

			print("test features saved")

	elif g_run_mode == 4:
		cnn.test(1, g_verbose, g_test_save_output, [], g_external_dir)


	tf.Session().close()
	for ds_ind in range(3):
		if g_input_hds[ds_ind] is not None:
			g_input_hds[ds_ind].close()

if __name__ == '__main__':
	main()

