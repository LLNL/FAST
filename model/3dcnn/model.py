################################################################################
# Copyright 2019-2021 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# File utility functions
################################################################################

import os
import sys
import math
import numbers
import numpy as np
import scipy as sp
import torch
import torch.nn as nn



def strip_prefix_if_present(state_dict, prefix):
	keys = sorted(state_dict.keys())
	if not all(len(key) == 0 or key.startswith(prefix) for key in keys):
		return

	for key in keys:
		newkey = key[len(prefix) :]
		state_dict[newkey] = state_dict.pop(key)

	try:
		metadata = state_dict._metadata
	except AttributeError:
		pass
	else:
		for key in list(metadata.keys()):
			if len(key) == 0:
				continue
			newkey = key[len(prefix) :]
			metadata[newkey] = metadata.pop(key)



class Model_3DCNN(nn.Module):

	# num_filters=[64,128,256] or [96,128,128]
	def __init__(self, feat_dim=19, output_dim=1, num_filters=[64,128,256], use_cuda=True, verbose=0):
		super(Model_3DCNN, self).__init__()
        
		self.feat_dim = feat_dim
		self.output_dim = output_dim
		self.num_filters = num_filters
		self.use_cuda = use_cuda
		self.verbose = verbose

		self.conv_block1 = self.__conv_layer_set__(self.feat_dim, self.num_filters[0], 7, 2, 3)
		self.res_block1 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)
		self.res_block2 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[0], 7, 1, 3)

		self.conv_block2 = self.__conv_layer_set__(self.num_filters[0], self.num_filters[1], 7, 3, 3)
		self.max_pool2 = nn.MaxPool3d(2)

		self.conv_block3 = self.__conv_layer_set__(self.num_filters[1], self.num_filters[2], 5, 2, 2)
		self.max_pool3 = nn.MaxPool3d(2)

		self.fc1 = nn.Linear(2048, 100)
		torch.nn.init.normal_(self.fc1.weight, 0, 1)
		self.fc1_bn = nn.BatchNorm1d(num_features=100, affine=True, momentum=0.1).train()
		self.fc2 = nn.Linear(100, 1)
		torch.nn.init.normal_(self.fc2.weight, 0, 1)
		self.relu = nn.ReLU()
		#self.drop=nn.Dropout(p=0.15)

	def __conv_layer_set__(self, in_c, out_c, k_size, stride, padding):
		conv_layer = nn.Sequential(
			nn.Conv3d(in_c, out_c, kernel_size=k_size, stride=stride, padding=padding, bias=True),
			nn.ReLU(inplace=True),
			nn.BatchNorm3d(out_c))
		return conv_layer

	def forward(self, x):
		if x.dim() == 1:
			x = x.unsqueeze(-1)
		conv1_h = self.conv_block1(x)
		if self.verbose != 0:
			print(conv1_h.shape)

		conv1_res1_h = self.res_block1(conv1_h)
		if self.verbose != 0:
			print(conv1_res1_h.shape)

		conv1_res1_h2 = conv1_res1_h + conv1_h
		if self.verbose != 0:
			print(conv1_res1_h2.shape)

		conv1_res2_h = self.res_block2(conv1_res1_h2)
		if self.verbose != 0:
			print(conv1_res2_h.shape)

		conv1_res2_h2 = conv1_res2_h + conv1_h
		if self.verbose != 0:
			print(conv1_res2_h2.shape)

		conv2_h = self.conv_block2(conv1_res2_h2)
		if self.verbose != 0:
			print(conv2_h.shape)

		pool2_h = self.max_pool2(conv2_h)
		if self.verbose != 0:
			print(pool2_h.shape)

		conv3_h = self.conv_block3(pool2_h)
		if self.verbose != 0:
			print(conv3_h.shape)

		pool3_h = conv3_h
		#pool3_h = self.max_pool3(conv3_h)
		#if self.verbose != 0:
		#	print(pool3_h.shape)

		flatten_h = pool3_h.view(pool3_h.size(0), -1)
		if self.verbose != 0:
			print(flatten_h.shape)

		fc1_z = self.fc1(flatten_h)
		fc1_y = self.relu(fc1_z)
		fc1_h = self.fc1_bn(fc1_y) if fc1_y.shape[0]>1 else fc1_y  #batchnorm train require more than 1 batch
		if self.verbose != 0:
			print(fc1_h.shape)

		fc2_z = self.fc2(fc1_h)
		if self.verbose != 0:
			print(fc2_z.shape)

		return fc2_z, fc1_z
