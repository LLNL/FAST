################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Fusion model of 3D CNN and GCN (or different 3D CNNs) for regression
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

sys.stdout.flush()
sys.path.insert(0, '../3dcnn')

from data_reader import *
from dnn_general import *


parser = argparse.ArgumentParser()
parser.add_argument("--main-dir", default=[], nargs="+", help="main model/dataset directory")
parser.add_argument("--fusionmodel-subdir", default=[], nargs="+", help="subdirectory storing fusion models/results (under main_dir)")
#parser.add_argument("--indmodel-subdirs", default=[], nargs="+", help="subdirectory storing individual models/results (under main_dir)")
#parser.add_argument("--csvfile", default=[], nargs="+", help="")
#parser.add_argument("--train-featfiles", default=[], nargs="+", help="")
#parser.add_argument("--val-featfiles", default=[], nargs="+", help="")
#parser.add_argument("--test-featfiles", default=[], nargs="+", help="")
parser.add_argument("--run-mode", default=[], nargs="+", help="1: training, 2: test, 3: test external testset and save features")
parser.add_argument("--external-dir", default=[], nargs="+")
parser.add_argument("--external-csvfile", default=[], nargs="+")
parser.add_argument("--external-3dcnn-featfile", default=[], nargs="+")
parser.add_argument("--external-sgcnn-featfile", default=[], nargs="+")
parser.add_argument("--external-outprefix", default=[], nargs="+")
args = parser.parse_args()


g_main_dir = '../../data'
g_model_subdirs = ['pdbbind2016_general_refined_sgcn_20190728', 'pdbbind2016_refined_pybel_processed_crystal_48_radius1_sigma1_rot0_model_3dcnn_res_result_20191009']

g_fusion_model_subdir = 'pdbbind2016_fusion_48_sgcn_20191009k'

g_train_feat_files = [['pybel_processed_pdbbind_2016_general_refined_Jul_28_19_18_12_1564362743_general_train_hidden_features.npy', 'pybel_processed_pdbbind_2016_general_refined_Jul_28_19_18_12_1564362743_refined_train_hidden_features.npy'], ['pdbbind2016_refined_pybel_processed_crystal_48_radius1_sigma1_rot0_model_3dcnn_res_result_20191009_general_train_fc10.npy', 'pdbbind2016_refined_pybel_processed_crystal_48_radius1_sigma1_rot0_model_3dcnn_res_result_20191009_train_fc10.npy']]
g_val_feat_files = [['pybel_processed_pdbbind_2016_general_refined_Jul_28_19_18_12_1564362743_general_val_hidden_features.npy', 'pybel_processed_pdbbind_2016_general_refined_Jul_28_19_18_12_1564362743_refined_val_hidden_features.npy'], ['pdbbind2016_refined_pybel_processed_crystal_48_radius1_sigma1_rot0_model_3dcnn_res_result_20191009_general_val_fc10.npy', 'pdbbind2016_refined_pybel_processed_crystal_48_radius1_sigma1_rot0_model_3dcnn_res_result_20191009_val_fc10.npy']]
g_test_feat_files = ['pybel_processed_pdbbind_2016_general_refined_Jul_28_19_18_12_1564362743_hidden_features.npy', 'pdbbind2016_refined_pybel_processed_crystal_48_radius1_sigma1_rot0_model_3dcnn_res_result_20191009_test_fc10.npy']

g_csv_file = 'pdbbind2016_general+refined_pybel_processed_crystal_48_radius1_sigma1_rot0_info.csv'

g_external_dir = ''
g_external_csv_file = ''
g_external_feat_files = ['', '']
g_external_out_prefix = ''

#g_model_subdirs = ['pdbbind_2018_with_water', 'pdbbind_2018_with_water']
#g_model_subdirs = ['pdbbind_2018_without_water', 'pdbbind_2018_without_water']
#g_test_feat_files = ['core_test_hidden_fc12.npy', 'pdbbind2016_refined_pybel_processed_crystal_48_radius1_sigma1_rot0_model_3dcnn_res_result_20191009_test_fc10.npy']
#g_csv_file = g_model_subdirs[0], 'pybel_processed_docking_48_radius1_sigma1_rot0_info.csv'

g_csv_ind_input = 1
g_csv_ind_output = 2
g_csv_ind_split = 3

g_input_dim = [48, 48, 48, 75]
g_input_type = 1
g_output_dim = [1]
g_output_type = 10

g_run_mode = 2 # 1: training, 2: testing, 3: testing on external test only data
g_epoch_count = 1000
g_batch_size = 100 # 50
g_save_rate = 0
g_optimizer_info = [1, 0.002, 0.9, 0.999, 1e-08]
g_decay_info = [1, 1000, 0.99]
g_loss_info = [2, 0, 0, 5e-2] # 1: L1, 2: L2 -> doesn't affect




def model_fusion_3(model_name, feat1, feat2, feat3, train_mode, reuse):
	with tf.variable_scope(model_name, reuse=reuse):
		dropout=0.5
	
		print(feat1.shape)
		fc11_w = weight_var([int(feat1.shape[1]), 10], stddev=0.01, name="fc11_w")
		fc11_b = bias_var([10], name="fc11_b")
		fc11_z = tf.matmul(feat1, fc11_w) + fc11_b
		fc11_h = lrelu(bn(tf.nn.dropout(fc11_z, keep_prob=dropout), train_mode,"fc11_bn"))
		print(fc11_h.shape)

		print(feat2.shape)
		fc12_w = weight_var([int(feat2.shape[1]), 10], stddev=0.01, name="fc12_w")
		fc12_b = bias_var([10], name="fc12_b")
		fc12_z = tf.matmul(feat2, fc12_w) + fc12_b
		fc12_h = lrelu(bn(tf.nn.dropout(fc12_z, keep_prob=dropout), train_mode,"fc12_bn"))
		print(fc12_h.shape)

		print(feat3.shape)
		fc13_w = weight_var([int(feat3.shape[1]), 10], stddev=0.01, name="fc13_w")
		fc13_b = bias_var([10], name="fc13_b")
		fc13_z = tf.matmul(feat3, fc13_w) + fc13_b
		fc13_h = lrelu(bn(tf.nn.dropout(fc13_z, keep_prob=dropout), train_mode,"fc13_bn"))
		print(fc13_h.shape)
	
		concat = tf.concat([fc11_h, fc12_h, fc13_h], 1)
		print(concat.shape)

		fc2_w = weight_var([30, 10], stddev=0.01, name="fc2_w")
		fc2_b = bias_var([10], name="fc2_b")
		fc2_z = tf.matmul(concat, fc2_w) + fc2_b
		fc2_h = bn(tf.nn.relu(fc2_z), train_mode,"fc2_bn")
		print(fc2_h.shape)
		
		fc3_w = weight_var([10, 1], stddev=0.01, name="fc3_w")
		fc3_b = bias_var([1], name="fc3_b")
		fc3_z = tf.matmul(fc2_h, fc3_w) + fc3_b
		print(fc3_z.shape)
	
	return fc3_z


def model_fusion_2(model_name, feat1, feat2, train_mode, reuse):
	with tf.variable_scope(model_name, reuse=reuse):
		#dropout=0.5
		dropout=1.0
	
		print(feat1.shape)
		fc11_w = weight_var([int(feat1.shape[1]), 5], stddev=0.01, name="fc11_w")
		fc11_b = bias_var([5], name="fc11_b")
		fc11_z = tf.matmul(feat1, fc11_w) + fc11_b
		fc11_h = lrelu(bn(tf.nn.dropout(fc11_z, keep_prob=dropout), train_mode,"fc11_bn"))
		print(fc11_h.shape)

		print(feat2.shape)
		fc12_w = weight_var([int(feat2.shape[1]), 5], stddev=0.01, name="fc12_w")
		fc12_b = bias_var([5], name="fc12_b")
		fc12_z = tf.matmul(feat2, fc12_w) + fc12_b
		fc12_h = lrelu(bn(tf.nn.dropout(fc12_z, keep_prob=dropout), train_mode,"fc12_bn"))
		print(fc12_h.shape)

		#fc2_h = fc11_h + fc12_h #-> becomes worse!

		concat = tf.concat([feat1, feat2, fc11_h, fc12_h], 1)
		#concat = tf.concat([feat1, fc11_h, fc12_h], 1)
		#concat = tf.concat([fc11_h, fc12_h], 1)
		print(concat.shape)

		fc2_w = weight_var([32, 10], stddev=0.01, name="fc2_w")
		fc2_b = bias_var([10], name="fc2_b")
		fc2_z = tf.matmul(concat, fc2_w) + fc2_b
		fc2_h = bn(tf.nn.relu(fc2_z), train_mode,"fc2_bn")
		print(fc2_h.shape)
		
		fc3_w = weight_var([10, 1], stddev=0.01, name="fc3_w")
		fc3_b = bias_var([1], name="fc3_b")
		fc3_z = tf.matmul(fc2_h, fc3_w) + fc3_b
		print(fc3_z.shape)
		
	return fc3_z


def load_data(feat_dir, feat_files):
	data = [np.load(os.path.join(feat_dir, feat_file)) for feat_file in feat_files]
	return np.concatenate(data, axis=0)


def main():
	if args.main_dir:
		g_main_dir = args.main_dir[0]
	if args.fusionmodel_subdir:
		g_fusion_model_subdir = args.fusionmodel_subdir[0]
	#if args.indmodel_subdirs:
	#	g_model_subdirs = args.indmodel_subdirs
	#if args.csvfile:
	#	g_csv_file = args.csvfile[0]
	#if args.train_featfiles:
	#	g_train_feat_files = args.train_featfiles # need to fix!!!
	#if args.val_featfiles:
	#	g_val_feat_files = args.val_featfiles # need to fix!!!
	#if args.test_featfiles:
	#	g_test_feat_files = args.test_featfiles # need to fix!!!
	if args.run_mode:
		g_run_mode = int(args.run_mode[0])
		print(g_run_mode)
	if args.external_dir:
		g_external_dir = args.external_dir[0]
	if args.external_csvfile:
		g_external_csv_file = args.external_csvfile[0]
	if args.external_3dcnn_featfile:
		g_external_feat_files[1] = args.external_3dcnn_featfile[0]
	if args.external_sgcnn_featfile:
		g_external_feat_files[0] = args.external_sgcnn_featfile[0]
	if args.external_outprefix:
		g_external_out_prefix = args.external_outprefix[0]

	# load dataset
	if g_run_mode == 3:
		data_reader = DataReader(g_external_dir, g_external_csv_file, g_csv_ind_input, g_csv_ind_output, g_csv_ind_split, g_input_dim, g_input_type, None, g_output_dim, g_output_type, None)
	else:
		data_reader = DataReader(g_main_dir, g_csv_file, g_csv_ind_input, g_csv_ind_output, g_csv_ind_split, g_input_dim, g_input_type, None, g_output_dim, g_output_type, None)
		
	train_count = len(data_reader.train_list)
	val_count = len(data_reader.val_list)
	test_count = len(data_reader.test_list)
	fusion_model_dir = os.path.join(g_main_dir, g_fusion_model_subdir)
	if not os.path.exists(fusion_model_dir):
		os.makedirs(fusion_model_dir)
	
	# load feature files
	if g_run_mode == 3:
		x_test = [np.load(os.path.join(g_external_dir, feat_file)) for feat_file in g_external_feat_files]
	else:
		x_train = [load_data(os.path.join(g_main_dir, model_subdir), feat_files) for model_subdir, feat_files in zip(g_model_subdirs, g_train_feat_files)]
		y_train = np.ndarray(shape=(train_count, 1), dtype=np.float32)
		for ind in range(train_count):
			input_info, output_info = data_reader.train_list[ind]
			y_train[ind] = float(output_info)
		x_val = [load_data(os.path.join(g_main_dir, model_subdir), feat_files) for model_subdir, feat_files in zip(g_model_subdirs, g_val_feat_files)]
		y_val = np.ndarray(shape=(val_count, 1), dtype=np.float32)
		for ind in range(val_count):
			input_info, output_info = data_reader.val_list[ind]
			y_val[ind] = float(output_info)

		x_test = [np.load(os.path.join(g_main_dir, model_subdir, feat_file)) for model_subdir, feat_file in zip(g_model_subdirs, g_test_feat_files)]

	y_test = np.ndarray(shape=(test_count, 1), dtype=np.float32)
	for ind in range(test_count):
		input_info, output_info = data_reader.test_list[ind]
		y_test[ind] = float(output_info)


	############################################################################
	# initialize fusion model
	# reset tf variables
	tf.reset_default_graph()
	
	# setup place holder for input, output
	input_ph1 = tf.placeholder(tf.float32, (None, x_test[0].shape[1]))
	input_ph2 = tf.placeholder(tf.float32, (None, x_test[1].shape[1]))
	#input_ph3 = tf.placeholder(tf.float32, (None, x_test[2].shape[1]))
	output_ph = tf.placeholder(tf.float32, (None, 1))
	training_phase_ph = tf.placeholder(tf.bool, name='training_phase_placeholder')
	#logit_ph = model_fusion_3('model_fusion_3', input_ph1, input_ph2, input_ph3, training_phase_ph, reuse=False)
	logit_ph = model_fusion_2('model_fusion_2', input_ph1, input_ph2, training_phase_ph, reuse=False)
		
	# setup loss
	loss = tf.reduce_sum(tf.reduce_mean(tf.square(tf.subtract(logit_ph, output_ph)), axis=0))
	tf.summary.scalar('loss', loss)
		
	# setup learning rate and decay
	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(g_optimizer_info[1], global_step, g_decay_info[1], g_decay_info[2], staircase=True)
	
	# setup optimizer
	optimizer = tf.train.AdamOptimizer(learning_rate, beta1=g_optimizer_info[2], beta2=g_optimizer_info[3], epsilon=g_optimizer_info[4]).minimize(loss, global_step=global_step)

	# for tensorboard
	merge_summary = tf.summary.merge_all()


	############################################################################
	# train/test fusion model
	if g_run_mode == 1:
		# start session
		saver = tf.train.Saver()
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.9
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			
			# load saved model if available
			ckpt = tf.train.get_checkpoint_state(fusion_model_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print("### checkpoint found -> model restored!")
		
			train_writer = tf.summary.FileWriter(fusion_model_dir, sess.graph)
			output_train_results = []
			output_val_results = []
			
			for epoch_ind in range(g_epoch_count):
				print('epoch - %d/%d' % (epoch_ind+1, g_epoch_count))

				# training
				batch_count = train_count // g_batch_size
				train_inds = np.array(range(batch_count * g_batch_size))
				random.shuffle(train_inds)
				l_avg = 0
				for batch_ind in range(batch_count):
					ind0 = batch_ind * g_batch_size
					ind1 = (batch_ind + 1) * g_batch_size
					inds = np.array(train_inds[ind0:ind1])
					x_batch1 = x_train[0][inds,:]
					x_batch2 = x_train[1][inds,:]
					#x_batch3 = x_train[2][inds,:]
					y_batch = y_train[inds]
					#feed_dict = {input_ph1: x_batch1, input_ph2: x_batch2, input_ph3: x_batch3, output_ph: y_batch, training_phase_ph : 1}
					feed_dict = {input_ph1: x_batch1, input_ph2: x_batch2, output_ph: y_batch, training_phase_ph : 1}
					_, l, lr, y_batch_pred, sstr = sess.run([optimizer, loss, learning_rate, logit_ph, merge_summary], feed_dict=feed_dict)
					print('[Training] [%d-%d]-[%d-%d] mini-batch loss: %f, learning rate: %f' % (epoch_ind+1, g_epoch_count, batch_ind+1, batch_count, l, lr))
				
					l_avg += l
					if g_save_rate > 0 and (batch_ind % g_save_rate == 0 or batch_ind == batch_count - 1):
						model_file = "model_%03d_%05d.ckpt" % (epoch_ind+1, batch_ind+1)
						model_path = os.path.join(fusion_model_dir, model_file)
						saver.save(sess, model_path)
						print('model saved: %s' % model_path)
					
					if batch_ind % 10 == 1:
						train_writer.add_summary(sstr, epoch_ind)

				# training epoch summary (loss, etc)
				l_avg = l_avg / batch_count
				output_train_results.append(l_avg)
				
				# validation
				batch_count = val_count // 1
				y_label = np.ndarray(shape=(batch_count * 1), dtype=np.float32)
				y_pred  = np.ndarray(shape=(batch_count * 1), dtype=np.float32)
				y_error = np.ndarray(shape=(batch_count * 1), dtype=np.float32)
				l_avg = 0
				for batch_ind in range(batch_count):
					ind0 = batch_ind * 1
					ind1 = (batch_ind + 1) * 1
					x_batch1 = x_val[0][ind0:ind1,:]
					x_batch2 = x_val[1][ind0:ind1,:]
					#x_batch3 = x_val[2][ind0:ind1,:]
					y_batch = y_val[ind0:ind1]
					#feed_dict = {input_ph1: x_batch1, input_ph2: x_batch2, input_ph3: x_batch3, output_ph: y_batch, training_phase_ph : 0}
					feed_dict = {input_ph1: x_batch1, input_ph2: x_batch2, output_ph: y_batch, training_phase_ph : 0}
					l, y_batch_pred = sess.run([loss, logit_ph], feed_dict=feed_dict)
					print('[Validating] [%d-%d]' % (batch_ind+1, batch_count))
					y_label[ind0:ind1] = y_batch[:,0]
					y_pred[ind0:ind1]  = y_batch_pred[:,0]
					y_error[ind0:ind1] = np.linalg.norm(y_batch - y_batch_pred, axis=1)
					l_avg += l

				# validation epoch summary
				l_avg = l_avg / batch_count
				output_val_results.append(l_avg)
				
				l2 = np.mean(y_error)
				rmse = math.sqrt(mean_squared_error(y_label, y_pred))
				mae = mean_absolute_error(y_label, y_pred)
				r2 = r2_score(y_label, y_pred)
				pearson, ppval = pearsonr(y_label, y_pred)
				spearman, spval = spearmanr(y_label, y_pred)
				mean = np.mean(y_pred)
				std = np.std(y_pred)
				print('[Validating] L2 error: %.3f, RMSE: %.3f, MAE: %.3f, R^2 score: %.3f, Pearson: %.3f, Spearman: %.3f, mean/std: %.3f/%.3f' % (l2, rmse, mae, r2, pearson, spearman, mean, std))

				if len(output_val_results) > 2 and l_avg <= min(np.asarray(output_val_results)):
					model_file = "model_%03d.ckpt" % (epoch_ind+1)
					model_path = os.path.join(fusion_model_dir, model_file)
					saver.save(sess, model_path)
					print('model saved: %s' % model_path)

			output_train_results_file = "output_train_summary.txt"
			with open(os.path.join(fusion_model_dir, output_train_results_file), 'w') as output_fp:
				for ind, loss in enumerate(output_train_results):
					out_str = '%3d %10.4f' % (ind+1, loss)
					output_fp.write(out_str)
					output_fp.write('\n')
			output_val_results_file = "output_val_summary.txt"
			with open(os.path.join(fusion_model_dir, output_val_results_file), 'w') as output_fp:
				for ind, loss in enumerate(output_val_results):
					out_str = '%3d %10.4f' % (ind+1, loss)
					output_fp.write(out_str)
					output_fp.write('\n')

	elif g_run_mode == 2 or g_run_mode == 3:
		# start session
		saver = tf.train.Saver()
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.9
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			
			# load saved model if available
			ckpt = tf.train.get_checkpoint_state(fusion_model_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print("### checkpoint found -> model restored!")
		
			batch_count = test_count // 1
			y_label = np.ndarray(shape=(batch_count * 1), dtype=np.float32)
			y_pred  = np.ndarray(shape=(batch_count * 1), dtype=np.float32)
			y_error = np.ndarray(shape=(batch_count * 1), dtype=np.float32)
			for batch_ind in range(batch_count):
				ind0 = batch_ind * 1
				ind1 = (batch_ind + 1) * 1
				x_batch1 = x_test[0][ind0:ind1,:]
				x_batch2 = x_test[1][ind0:ind1,:]
				#x_batch3 = x_test[2][ind0:ind1,:]
				y_batch = y_test[ind0:ind1]
				#feed_dict = {input_ph1: x_batch1, input_ph2: x_batch2, input_ph3: x_batch3, training_phase_ph : 0}
				feed_dict = {input_ph1: x_batch1, input_ph2: x_batch2, training_phase_ph : 0}
				y_batch_pred = sess.run(logit_ph, feed_dict=feed_dict)
				print('[Testing] [%d-%d]' % (batch_ind+1, batch_count))
				y_label[ind0:ind1] = y_batch[:,0]
				y_pred[ind0:ind1]  = y_batch_pred[:,0]
				y_error[ind0:ind1] = np.linalg.norm(y_batch - y_batch_pred, axis=1)

			l2 = np.mean(y_error)
			rmse = math.sqrt(mean_squared_error(y_label, y_pred))
			mae = mean_absolute_error(y_label, y_pred)
			r2 = r2_score(y_label, y_pred)
			pearson, ppval = pearsonr(y_label, y_pred)
			spearman, spval = spearmanr(y_label, y_pred)
			mean = np.mean(y_pred)
			std = np.std(y_pred)
			print('[Testing] L2 error: %.3f, RMSE: %.3f, MAE: %.3f, R^2 score: %.3f, Pearson: %.3f, Spearman: %.3f, mean/std: %.3f/%.3f' % (l2, rmse, mae, r2, pearson, spearman, mean, std))

			if g_run_mode == 3:
				output_dir = g_external_dir
				output_file = "%s_pred.txt" % g_external_out_prefix
			else:
				output_dir = fusion_model_dir
				output_file = "fusion_output_test_pred.txt"

			np.save(os.path.join(output_dir, output_file[:-3] + 'npy'), y_pred)
			with open(os.path.join(output_dir, output_file), 'w') as output_fp:
				for out_ind in range(y_error.shape[0]):
					out_str = '%10.4f %10.4f %10.4f' % (y_label[out_ind], y_pred[out_ind], y_error[out_ind])
					output_fp.write(out_str)
					output_fp.write('\n')


	tf.Session().close()


if __name__ == '__main__':
	main()

