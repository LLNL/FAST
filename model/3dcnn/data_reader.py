################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Dataset reader interface
################################################################################


from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import scipy as sp
import pandas as pd
import random
import itertools
import csv
import h5py

from file_util import *

# note:
# input_type/output_type: 0: use external reader functions (base_dir, file_name), 1: npy, 2: image file (jpg, png, etc), 3: ndarray/volume (raw, vol, npfile), 10: use the number/class in the csv (for classification, regression)
# csv_ind_output: label for classification

#g_data_type_default_ext = ['', 'npy', 'png', 'raw', '', '', '', '', '', '', 'txt']



class DataReader:
	def __init__(self, main_dir, csv_fname, csv_ind_input, csv_ind_output, csv_ind_split, input_dim, input_type, input_reader, output_dim, output_type, output_reader, num_classes=-1, class_balanced=False, class_ratio=1.5):
		self.main_dir = main_dir
		self.input_dim = input_dim
		self.input_type = input_type
		self.input_reader = input_reader
		self.output_dim = output_dim
		self.output_type = output_type
		self.output_reader = output_reader
		self.num_classes = num_classes # for classification problem, > 0: classification
		self.class_balanced = class_balanced
		self.class_max_ratio = class_ratio
		
		self.batch_input_dim = [0 for n in range(len(self.input_dim)+1)]
		for ind, dim in enumerate(self.input_dim):
			self.batch_input_dim[ind+1] = dim

		self.batch_output_dim = [0 for n in range(len(self.output_dim)+1)]
		for ind, dim in enumerate(self.output_dim):
			self.batch_output_dim[ind+1] = dim

		self.train_list = []
		self.val_list = []
		self.test_list = []
		
		if valid_file(os.path.join(main_dir, csv_fname)):
			with open(os.path.join(main_dir, csv_fname), 'r') as csv_file:
				first_line = csv_file.readlines()[0]
				sniffer = csv.Sniffer()
				has_header = sniffer.has_header(first_line)
			
				csv_file.seek(0)
				csv_reader = csv.reader(csv_file)
				if has_header:
					next(csv_reader)
				for row in csv_reader:
					data_input_info = row[csv_ind_input] # relative path from the main_dir!!!
					data_output_info = row[csv_ind_output] # relative path from the main_dir or the actual label (class/number)
					data_split = int(row[csv_ind_split])
					if data_split == 0:
						self.train_list.append([data_input_info, data_output_info])
					elif data_split == 1:
						self.val_list.append([data_input_info, data_output_info])
					else:
						self.test_list.append([data_input_info, data_output_info])

		self.train_batch_ind = 0
		self.train_batch_size = 0
		self.train_batch_count = 0

		self.val_batch_ind = 0
		self.val_batch_size = 0
		self.val_batch_count = 0

		self.test_batch_ind = 0
		self.test_batch_size = 0
		self.test_batch_count = 0
			
		self.train_list_balanced = []
		self.class_max_counts = 0
		if self.class_balanced:
			train_hist, _ = self.get_class_balance_info()
			self.class_max_counts = int(float(np.min(train_hist)) * self.class_max_ratio)

	def __load_input__(self, ind, input_info):
		if self.input_type == 0:
			data = self.input_reader(ind, self.main_dir, input_info)
		elif self.input_type == 1:
			data = np.load(os.path.join(self.main_dir, input_info))
			if input_info[-1] == 'z': # in case of compressed npz
				data = data["vol_data"]
		elif self.input_type == 2:
			data = sp.misc.imread(os.path.join(self.main_dir, input_info))
			data = data.reshape(self.input_dim)
		elif self.input_type == 3:
			data = np.fromfile(os.path.join(self.main_dir, input_info), dtype=np.float32)
			data = data.reshape(self.input_dim)
		elif self.input_type == 10:
			data = float(input_info)
		return data

	def __load_output__(self, ind, output_info):
		if self.output_type == 0:
			data = self.output_reader(ind, self.main_dir, output_info)
		elif self.output_type == 1:
			data = np.load(os.path.join(self.main_dir, output_info))
		elif self.output_type == 2:
			data = sp.misc.imread(os.path.join(self.main_dir, output_info))
			data = data.reshape(self.output_dim)
		elif self.output_type == 3:
			data = np.fromfile(os.path.join(self.main_dir, output_info), dtype = np.float32)
			data = data.reshape(self.output_dim)
		elif self.output_type == 10:
			if self.num_classes > 0:
				data = int(output_info)
			else:
				data = float(output_info)
		return data

	def __save_input__(self, input, input_path): # note that it doesn't use main_dir
		if self.input_type == 0:
			print('not supported')
		elif self.input_type == 1:
			np.save(input_path, input)
		elif self.input_type == 2:
			if len(input.shape) == 3 and input.shape[2] == 1:
				input = input.reshape((input.shape[0], input.shape[1]))
			sp.misc.imsave(input_path, input)
		elif self.input_type == 3:
			input.tofile(input_path)
	
	def __save_output__(self, output, output_path): # note that it doesn't use main_dir
		if self.output_type == 0:
			print('not supported')
		elif self.output_type == 1:
			np.save(output_path, output)
		elif self.output_type == 2:
			if len(output.shape) == 3 and output.shape[2] == 1:
				output = output.reshape((output.shape[0], output.shape[1]))
			sp.misc.imsave(output_path, output)
		elif self.output_type == 3:
			output.tofile(output_path)

	def begin_train(self, batch_size, shuffle=True):
		self.train_batch_ind = 0
		self.train_batch_size = batch_size
		self.batch_input_dim[0] = self.train_batch_size
		self.batch_output_dim[0] = self.train_batch_size
		self.input_batch  = np.zeros(self.batch_input_dim, dtype=np.float32)
		if self.num_classes > 0:
			self.output_batch = np.zeros((self.train_batch_size), dtype=np.int32)
		else:
			self.output_batch = np.zeros(self.batch_output_dim, dtype=np.float32)
		if shuffle:
			random.shuffle(self.train_list)
		
		if self.class_balanced:
			self.train_list_balanced = []
			class_count = [0 for n in range(self.num_classes)]
			for input_info, output_info in self.train_list:
				label = int(output_info)
				if class_count[label] > self.class_max_counts:
					continue
				self.train_list_balanced.append([input_info, output_info])
				class_count[label] += 1
			random.shuffle(self.train_list_balanced)
			self.train_batch_count = int(len(self.train_list_balanced) // batch_size)
		else:
			self.train_batch_count = int(len(self.train_list) // batch_size)

	def next_train(self):
		for n in range(self.train_batch_size):
			ind = n + self.train_batch_ind * self.train_batch_size
			if self.class_balanced:
				input_info, output_info = self.train_list_balanced[ind]
			else:
				input_info, output_info = self.train_list[ind]
			self.input_batch[n] = self.__load_input__(ind, input_info)
			self.output_batch[n] = self.__load_output__(ind, output_info)
		
		self.train_batch_ind = (self.train_batch_ind+1) % self.train_batch_count
		return self.input_batch, self.output_batch

	def begin_val(self, batch_size):
		self.val_batch_ind = 0
		self.val_batch_size = batch_size
		self.val_batch_count = int(len(self.val_list) // batch_size)
		self.batch_input_dim[0] = self.val_batch_size
		self.batch_output_dim[0] = self.val_batch_size
		
		self.input_batch  = np.zeros(self.batch_input_dim, dtype=np.float32)
		if self.num_classes > 0:
			self.output_batch = np.zeros((self.val_batch_size), dtype=np.int32)
		else:
			self.output_batch = np.zeros(self.batch_output_dim, dtype=np.float32)
	
	def next_val(self):
		for n in range(self.val_batch_size):
			ind = n + self.val_batch_ind * self.val_batch_size
			input_info, output_info = self.val_list[ind]
			self.input_batch[n] = self.__load_input__(ind, input_info)
			self.output_batch[n] = self.__load_output__(ind, output_info)
		
		self.val_batch_ind = (self.val_batch_ind+1) % self.val_batch_count
		return self.input_batch, self.output_batch

	def begin_test(self, batch_size):
		self.test_batch_ind = 0
		self.test_batch_size = batch_size
		self.test_batch_count = int(len(self.test_list) // batch_size)
		self.batch_input_dim[0] = self.test_batch_size
		self.batch_output_dim[0] = self.test_batch_size
		
		self.input_batch  = np.zeros(self.batch_input_dim, dtype=np.float32)
		if self.num_classes > 0:
			self.output_batch = np.zeros((self.test_batch_size), dtype=np.int32)
		else:
			self.output_batch = np.zeros(self.batch_output_dim, dtype=np.float32)
	
	def next_test(self):
		for n in range(self.test_batch_size):
			ind = n + self.test_batch_ind * self.test_batch_size
			input_info, output_info = self.test_list[ind]
			self.input_batch[n] = self.__load_input__(ind, input_info)
			self.output_batch[n] = self.__load_output__(ind, output_info)
		
		self.test_batch_ind = (self.test_batch_ind+1) % self.test_batch_count
		return self.input_batch, self.output_batch

	def get_test(self, test_inds):
		indim = [0 for n in range(len(self.input_dim)+1)]
		indim[0] = len(test_inds)
		for ind, dim in enumerate(self.input_dim):
			indim[ind+1] = dim

		outdim = [0 for n in range(len(self.output_dim)+1)]
		outdim[0] = len(test_inds)
		for ind, dim in enumerate(self.output_dim):
			outdim[ind+1] = dim
	
		inbatch = np.zeros(indim, dtype=np.float32)
		if self.num_classes > 0:
			outbatch = np.zeros((len(test_inds)), dtype=np.int32)
		else:
			outbatch = np.zeros(outdim, dtype=np.float32)
		
		infobatch = []
		for n, ind in enumerate(test_inds):
			input_info, output_info = self.test_list[ind]
			inbatch[n] = self.__load_input__(ind, input_info)
			outbatch[n] = self.__load_output__(ind, output_info)
			infobatch.append([input_info, output_info])
		
		return inbatch, outbatch, infobatch

	def get_class_balance_info(self):
		if self.num_classes <= 0:
			return [], []
		train_data_hist = [0 for n in range(self.num_classes)]
		val_data_hist = [0 for n in range(self.num_classes)]
		test_data_hist = [0 for n in range(self.num_classes)]
		for (_, output_info) in self.train_list:
			train_data_hist[int(output_info)] += 1
		for (_, output_info) in self.val_list:
			val_data_hist[int(output_info)] += 1
		for (_, output_info) in self.test_list:
			test_data_hist[int(output_info)] += 1
		return train_data_hist, test_data_hist
















class MetricDataReader:
	def __init__(self, main_dir, csv_fname, csv_ind_input1, csv_ind_input2, csv_ind_label, csv_ind_split, input_dim, input_type, input_reader=None):
		a = 0 #########






#class Dataset_ATOM_Pair:
#	def __init__(self, main_dir='', train_subdir='', test_subdir='', input_dim=[64,64,64,8], label_mode=2):
#		self.train_dir = os.path.join(main_dir, train_subdir)
#		self.test_dir = os.path.join(main_dir, test_subdir)
#		self.num_classes = 2 # same or not (1, 0)
#		self.input_dim = input_dim
#		self.label_mode = label_mode
#		
#		self.train_pos_list = []
#		self.train_neg_list = []
#		self.train_batch_ind = 0
#		self.train_batch_size = 0
#		self.train_batch_count = 0
#		
#		self.test_pos_list = []
#		self.test_neg_list = []
#		self.test_batch_ind = 0
#		self.test_batch_size = 0
#		self.test_batch_count = 0
#	
#		file_list = get_files_ext(self.train_dir, 'vol')
#		for file_name in file_list:
#			if self.label_mode == 1:
#				if file_name.startswith('act'):
#					self.train_pos_list.append(file_name)
#				else:
#					self.train_neg_list.append(file_name)
#			elif self.label_mode == 2:
#				label = int(file_name[-5:-4])
#				if label == 0 or label == 1:
#					self.train_neg_list.append(file_name)
#				else:
#					self.train_pos_list.append(file_name)
#
#		file_list = get_files_ext(self.test_dir, 'vol')
#		for file_name in file_list:
#			if self.label_mode == 1:
#				if file_name.startswith('act'):
#					self.train_pos_list.append(file_name)
#				else:
#					self.train_neg_list.append(file_name)
#			elif self.label_mode == 2:
#				label = int(file_name[-5:-4])
#				if label == 0 or label == 1:
#					self.train_neg_list.append(file_name)
#				else:
#					self.train_pos_list.append(file_name)
#
#	def __load_vol__(self, vol_filename):
#		vol_data = np.fromfile(vol_filename, dtype = np.float32)
#		vol_data = vol_data.reshape([self.input_dim[0], self.input_dim[1], self.input_dim[2], self.input_dim[3]])
#		return vol_data
#
#	def begin_train(self, batch_size):
#		random.shuffle(self.train_pos_list)
#		random.shuffle(self.train_neg_list)
#		
#		self.train_pospair_list = []
#		self.train_pospair_list.extend(list(itertools.combinations(self.train_pos_list, 2))[:g_max_pair_count])
#		self.train_pospair_list.extend(list(itertools.combinations(self.train_neg_list, 2))[:g_max_pair_count])
#		
#		self.train_negpair_list = []
#		while len(self.train_negpair_list) < len(self.train_pospair_list):
#			for train_pos_item in self.train_pos_list:
#				self.train_negpair_list.append((train_pos_item, random.choice(self.train_neg_list)))
#			for train_neg_item in self.train_neg_list:
#				self.train_negpair_list.append((train_neg_item, random.choice(self.train_pos_list)))
#		self.train_negpair_list = self.train_negpair_list[:len(self.train_pospair_list)]
#
#		self.train_batch_ind = 0
#		self.train_batch_size = batch_size
#		self.train_batch_count = (len(self.train_pospair_list) * 2) / batch_size
#
#	def next_train(self):
#		vol_batch1 = np.zeros((self.train_batch_size, self.input_dim[0], self.input_dim[1], self.input_dim[2], self.input_dim[3]), dtype=np.float32)
#		vol_batch2 = np.zeros((self.train_batch_size, self.input_dim[0], self.input_dim[1], self.input_dim[2], self.input_dim[3]), dtype=np.float32)
#		label_batch = np.zeros((self.train_batch_size), dtype=np.int32)
#
#		noffset = self.train_batch_ind * self.train_batch_size / 2
#		for n in range(self.train_batch_size):
#			ind = (noffset + n) / 2
#			if n % 2 == 0:
#				(file_name1, file_name2) = self.train_pospair_list[ind]
#				label = 1
#			else:
#				(file_name1, file_name2) = self.train_negpair_list[ind]
#				label = 0
#			
#			vol_batch1[n,:,:,:,:] = self.__load_vol__(os.path.join(self.train_dir, file_name1))
#			vol_batch2[n,:,:,:,:] = self.__load_vol__(os.path.join(self.train_dir, file_name2))
#			label_batch[n] = label
#
#		self.train_batch_ind = (self.train_batch_ind+1) % self.train_batch_count
#		return vol_batch1, vol_batch2, label_batch
#
#	def begin_test(self, batch_size):
#		self.test_pospair_list = []
#		self.test_pospair_list.extend(list(itertools.combinations(self.test_pos_list, 2))[:g_max_pair_count])
#		self.test_pospair_list.extend(list(itertools.combinations(self.test_neg_list, 2))[:g_max_pair_count])
#		
#		self.test_negpair_list = []
#		while len(self.test_negpair_list) < len(self.test_pospair_list):
#			for test_pos_item in self.test_pos_list:
#				self.test_negpair_list.append((test_pos_item, random.choice(self.test_neg_list)))
#			for test_neg_item in self.test_neg_list:
#				self.test_negpair_list.append((test_neg_item, random.choice(self.test_pos_list)))
#		self.test_negpair_list = self.test_negpair_list[:len(self.test_pospair_list)]
#		
#		self.test_batch_ind = 0
#		self.test_batch_size = batch_size
#		self.test_batch_count = (len(self.test_pospair_list) + len(self.test_pospair_list)) / batch_size
#
#	def next_test(self):
#		vol_batch1 = np.zeros((self.test_batch_size, self.input_dim[0], self.input_dim[1], self.input_dim[2], self.input_dim[3]), dtype=np.float32)
#		vol_batch2 = np.zeros((self.test_batch_size, self.input_dim[0], self.input_dim[1], self.input_dim[2], self.input_dim[3]), dtype=np.float32)
#		label_batch = np.zeros((self.test_batch_size), dtype=np.int32)
#
#		noffset = self.test_batch_ind * self.test_batch_size
#		for n in range(self.test_batch_size):
#			ind = noffset + n
#			if ind < len(self.test_pospair_list):
#				(file_name1, file_name2) = self.test_pospair_list[ind]
#				label = 1
#			else:
#				(file_name1, file_name2) = self.test_negpair_list[ind - len(self.test_pospair_list)]
#				label = 0
#
#			vol_batch1[n,:,:,:,:] = self.__load_vol__(os.path.join(self.test_dir, file_name1))
#			vol_batch2[n,:,:,:,:] = self.__load_vol__(os.path.join(self.test_dir, file_name2))
#			label_batch[n] = label
#		
#		self.test_batch_ind = (self.test_batch_ind+1) % self.test_batch_count
#		return vol_batch1, vol_batch2, label_batch
#
#
#if __name__ == '__main__':
##	data_reader = Dataset_ATOM('/Users/kim63/LLNL/Data/ATOM/Try1_dataset_3d/', 'cnn_3d_180222_train', 'cnn_3d_180222_test')
##	data_reader.begin_train(50)
##	for batch_ind in range(data_reader.train_batch_count):
##		x_batch, y_batch = data_reader.next_train()
##		print('%d/%d' % (batch_ind, data_reader.train_batch_count))
##		print(x_batch.shape)
##		print(y_batch.shape)
##		print(y_batch)
##	data_reader.begin_test(50)
##	for batch_ind in range(data_reader.test_batch_count):
##		x_batch, y_batch = data_reader.next_test()
##		print(x_batch.shape)
##		print(y_batch.shape)
##		print(y_batch)
#
#	data_reader_pair = Dataset_ATOM_Pair('/Users/kim63/LLNL/Data/ATOM/Try1_dataset_3d/', 'cnn_3d_180222_train', 'cnn_3d_180222_test')
#	data_reader_pair.begin_train(100)
#	#for batch_ind in range(data_reader_pair.train_batch_count):
#	#	x_batch1, x_batch2, y_batch = data_reader_pair.next_train()
#	#	print('%d/%d' % (batch_ind, data_reader_pair.train_batch_count))
#	#	print(batch_ind)
#	#	print(x_batch1.shape)
#	#	print(x_batch2.shape)
#	#	print(y_batch)
#
#	data_reader_pair.begin_test(100)
#	for batch_ind in range(data_reader_pair.test_batch_count):
#		x_batch1, x_batch2, y_batch = data_reader_pair.next_test()
#		print('%d/%d' % (batch_ind, data_reader_pair.test_batch_count))
#		print(batch_ind)
#		print(x_batch1.shape)
#		print(x_batch2.shape)
#		print(y_batch)

