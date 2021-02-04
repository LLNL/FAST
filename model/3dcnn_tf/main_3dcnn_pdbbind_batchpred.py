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
from file_util import *

sys.stdout.flush()


g_input_3D_relative_size = True ## pdbbind: True, covid: False -> but it gives too low prediction values (2~3)
g_input_3D_size_angstrom = 48
g_input_3D_dim = 48  # 24, 32, 48
g_input_3D_atom_radius = 1
g_input_3D_atom_radii = False
g_input_3D_sigma = 1
g_input_3D_rotate = False  # above one can be applied only if this is true
g_input_dim = [g_input_3D_dim, g_input_3D_dim, g_input_3D_dim, 19]
g_input_type = 0
g_output_dim = [1]
g_output_type = 10

g_csv_header = ['ligand_id', 'file_prefix', 'label', 'train_test_split']
g_csv_suffix = "info.csv"
g_csv_ind_input = 1
g_csv_ind_output = 2
g_csv_ind_split = 3

g_optimizer_info = [1, 0.0007, 0.9, 0.999, 1e-08] # default is 0.0007
g_decay_info = [1, 100, 0.95]
g_loss_info = [2, 0, 0, 5e-2]  # 1: l1, 2: l2

g_input_hd = None

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", default="", help="directory storing models/results")
parser.add_argument("--data-dir", default="", help="directory storing input hdf files")
parser.add_argument("--input-type", default=2, help="2: general ML-HDF (docking), 3: general ML-HDF (crystal)")
parser.add_argument("--input-ext", default=".hdf")
parser.add_argument("--input-suffix", default="cut10.hdf")
parser.add_argument("--output-suffix", default="_3dcnn")
args = parser.parse_args()




def main():
	global g_input_hd

	# initialize CNN with a dummy data reader
	data_reader = DataReader(args.data_dir, "", g_csv_ind_input, g_csv_ind_output, g_csv_ind_split, g_input_dim, g_input_type, input_reader, g_output_dim, g_output_type, None)
	cnn = DNN_General(data_reader, model_3dcnn_res, "model_3dcnn_res", output_dir=args.model_dir, optimizer_info=g_optimizer_info, decay_info=g_decay_info, model_loss_info=g_loss_info)

	# start TF session
	sess, saver = cnn.begin_sess_custom()

	# loop over hdf files in a specified directory
	hdf_file_list = get_files_ext(args.data_dir, args.input_suffix)
	hdf_file_list.sort()
	for hdf_fn in hdf_file_list:
		output_prefix = hdf_fn[:-len(args.input_ext)] + args.output_suffix
		
		# skip if its log file exists
		logfile_path = os.path.join(args.data_dir, output_prefix + ".log")
		if valid_file(logfile_path):
			continue
		print("##### evaluating %s #####" % (hdf_fn))
		
		# create csv file and load hdf file
		csv_fn = output_prefix + ".csv"
		with open(os.path.join(args.data_dir, csv_fn), 'w') as csv_fp:
			csv_writer = csv.writer(csv_fp, delimiter=',')
			csv_writer.writerow(g_csv_header)

			g_input_hd = h5py.File(os.path.join(args.data_dir, hdf_fn), 'r', driver='core')
			#g_input_hd = h5py.File(os.path.join(args.data_dir, hdf_fn), 'r', chunk_cache_mem_size=1024**2*500)
			if g_input_hd == None:
				continue
				
			ds_ind = 2 # always test set
			input_type = int(args.input_type)
			for lig_id in g_input_hd.keys():
				if input_type == 2:
					pose_data = g_input_hd[lig_id]["pybel"]["processed"]["docking"]
					for pose_id in range(1,11):
						if not str(pose_id) in pose_data:
							continue
		
						lig_id_pose = lig_id + '_' + str(pose_id)
						lig_prefix = "%d/%s/%s" % (ds_ind, hdf_fn, lig_id_pose)
						lig_affinity = g_input_hd[lig_id].attrs["affinity"]
						csv_writer.writerow([lig_id_pose, lig_prefix, lig_affinity, ds_ind])
				else:
					lig_prefix = "%d/%s/%s" % (ds_ind, hdf_fn, lig_id)
					lig_affinity = g_input_hd[lig_id].attrs["affinity"]
					csv_writer.writerow([lig_id, lig_prefix, lig_affinity, ds_ind])

		# open data reader
		data_reader = DataReader(args.data_dir, csv_fn, g_csv_ind_input, g_csv_ind_output, g_csv_ind_split, g_input_dim, g_input_type, input_reader, g_output_dim, g_output_type, None)
		cnn.data_reader = data_reader

		# run evaluation and save features
		data_reader.begin_test(1)
		test_pred_arr = np.ndarray(shape=(data_reader.test_batch_count), dtype=np.float32)
		test_feat_arr1 = np.ndarray(shape=(data_reader.test_batch_count, 10), dtype=np.float32)
		test_feat_arr2 = np.ndarray(shape=(data_reader.test_batch_count, 256), dtype=np.float32)
		custom_vars = [test_pred_arr, test_feat_arr1, test_feat_arr2]
		cnn.test_sess_custom(sess, saver, 1, run_custom_batch, custom_vars)
		np.save(os.path.join(args.data_dir, '%s_pred.npy' % output_prefix), test_pred_arr)
		np.save(os.path.join(args.data_dir, '%s_fc10.npy' % output_prefix), test_feat_arr1)
		np.save(os.path.join(args.data_dir, '%s_fc256.npy' % output_prefix), test_feat_arr2)

		with open(logfile_path, 'w') as f:
			f.write("done\n")


	tf.Session().close()


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
	global g_input_hd

	input_type = int(args.input_type)
	if input_type == 2:
		#split = int(input_info.split('/')[0])
		lig_id_pose = input_info.split('/')[2]
		lig_id = '_'.join(lig_id_pose.split('_')[:-1])
		pose_id = lig_id_pose.split('_')[-1]
		#input_hd = g_input_hds[split]
		#input_data_ = g_input_hd[lig_id]["pybel"]["processed"]["docking"][pose_id]["data"]
		input_data_ = g_input_hd[lig_id]["pybel"]["processed"]["docking"][pose_id]["data"][:]
		
	elif input_type == 3:
		#split = int(input_info.split('/')[0])
		lig_id = input_info.split('/')[2]
		#input_hd = g_input_hds[split]
		#input_data_ = g_input_hd[lig_id]["pybel"]["processed"]["pdbbind"]["data"]
		input_data_ = g_input_hd[lig_id]["pybel"]["processed"]["pdbbind"]["data"][:]
		
	input_radii = None
	if g_input_3D_atom_radii:
		input_radii = input_data_.attrs['van_der_waals']
	input_xyz = input_data_[:,0:3]
	input_feat = input_data_[:,3:]
	xmin, ymin, zmin, xmax, ymax, zmax = get_3D_bound(input_xyz)
	input_data = get_3D_all2(input_xyz, input_feat, g_input_dim, g_input_3D_relative_size, g_input_3D_size_angstrom, \
					input_radii, g_input_3D_atom_radius, g_input_3D_sigma)

	return input_data


def get_3D_bound(xyz_array):
	xmin = min(xyz_array[:, 0])
	ymin = min(xyz_array[:, 1])
	zmin = min(xyz_array[:, 2])
	xmax = max(xyz_array[:, 0])
	ymax = max(xyz_array[:, 1])
	zmax = max(xyz_array[:, 2])
	return xmin, ymin, zmin, xmax, ymax, zmax


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


def model_3dcnn_res(model_name, img_data, train_mode, reuse):
	num_filters = [64, 128, 256]
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


if __name__ == '__main__':
	main()

