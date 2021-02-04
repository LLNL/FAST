################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# DNN/CNN-based learning for general use
################################################################################


from __future__ import absolute_import
from __future__ import print_function

import math
import numpy as np
import scipy as sp
import pandas as pd
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

from sklearn.metrics import *
from skimage.measure import compare_mse, compare_psnr, compare_ssim
from scipy.stats import *

from file_util import *
from metric_util import *
from eval_util import *
from tf_util import *
from data_reader import *

# model_loss_info = [mode, model_loss_function], mode 0: user-defined, mode 1: l1, mode 2: l2, mode 3: SSIM (for image), model 4: MS-SSIM (for image), model 5: L1 + MS-SSIM (for image)

MSSSIM_WEIGHTS2 = (0.6, 0.3, 0.1)




class DNN_General:
	def __init__(self, data_reader, model, model_name, model_dsc=None, model_dsc_name='', output_dir='', optimizer_info=[1, 0.001, 0.9, 0.999, 1e-08], decay_info=[0, 100000, 0.9], model_loss_info=[2, 0, 0, 5e-2], model_loss_function=None):
		self.data_reader = data_reader
		self.model = model
		self.model_name = model_name
		self.model_dsc = model_dsc
		self.model_dsc_name = model_dsc_name
		self.model_adv = False
		self.output_dir = output_dir

		# create output directory if not exist
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		# reset tf variables
		tf.reset_default_graph()

		# setup place holder for input, output
		input_dim = data_reader.batch_input_dim
		output_dim = data_reader.batch_output_dim
		input_dim[0] = None
		output_dim[0] = None
		self.input_ph = tf.placeholder(tf.float32, input_dim)
		if model_loss_info[0] >= 100 and model_loss_info[0] <= 110:
			self.output_ph = tf.placeholder(tf.int32, output_dim)
		else:
			self.output_ph = tf.placeholder(tf.float32, output_dim)
		self.training_phase_ph = tf.placeholder(tf.bool, name='training_phase_placeholder')
		self.training_epoch_ph = tf.placeholder(tf.int32, name='training_epoch_placeholder')
		self.logit_ph, self.user_phs = model(self.model_name, self.input_ph, self.training_phase_ph, reuse=False)
		if self.model_dsc and self.model_dsc_name:
			self.model_adv = True
			self.D_real_ph, self.D_real_logits_ph = model_dsc(self.model_dsc_name, self.output_ph, self.training_phase_ph)
			self.D_fake_ph, self.D_fake_logits_ph = model_dsc(self.model_dsc_name, self.logit_ph, self.training_phase_ph, reuse=True)

		# setup model loss
		if model_loss_info[0] == 0 and model_loss_function:
			self.model_loss, self.model_loss_each = model_loss_function(self.input_ph, self.output_ph, self.logit_ph, self.user_phs, self.training_epoch_ph)
		elif model_loss_info[0] == 1: # L1 loss
			self.model_loss_each = tf.reduce_sum(tf.abs(tf.subtract(self.logit_ph, self.output_ph)), axis=1)
			self.model_loss = tf.reduce_sum(tf.reduce_mean(tf.abs(tf.subtract(self.logit_ph, self.output_ph)), axis=0))
		elif model_loss_info[0] == 2: # L2 loss
			#self.model_loss = tf.nn.l2_loss(tf.subtract(self.logit_ph, self.output_ph))
			self.model_loss_each = tf.reduce_sum(tf.square(tf.subtract(self.logit_ph, self.output_ph)), axis=1)
			self.model_loss = tf.reduce_sum(tf.reduce_mean(tf.square(tf.subtract(self.logit_ph, self.output_ph)), axis=0))
		elif model_loss_info[0] == 3: # SSIM
			self.model_loss = (1 - tf.reduce_mean(tf.image.ssim(self.logit_ph, self.output_ph, model_loss_info[1])))
		elif model_loss_info[0] == 4: # MS-SSIM
			self.model_loss = (1 - tf.reduce_mean(tf.image.ssim_multiscale(self.logit_ph, self.output_ph, model_loss_info[1])))
		elif model_loss_info[0] == 11: # L1 + SSIM
			self.model_loss = tf.reduce_sum(tf.reduce_mean(tf.abs(tf.subtract(self.logit_ph, self.output_ph)), axis=0)) * model_loss_info[2]
			self.model_loss += (1 - tf.reduce_mean(tf.image.ssim(self.logit_ph, self.output_ph, model_loss_info[1]))) * (1 - model_loss_info[2])
		elif model_loss_info[0] == 12: # L2 + SSIM
			self.model_loss = tf.reduce_sum(tf.reduce_mean(tf.square(tf.subtract(self.logit_ph, self.output_ph)), axis=0)) * model_loss_info[2]
			self.model_loss += (1 - tf.reduce_mean(tf.image.ssim(self.logit_ph, self.output_ph, model_loss_info[1]))) * (1 - model_loss_info[2])
		elif model_loss_info[0] == 21: # L1 + MS-SSIM
			self.model_loss = tf.reduce_sum(tf.reduce_mean(tf.abs(tf.subtract(self.logit_ph, self.output_ph)), axis=0)) * model_loss_info[2]
			self.model_loss += (1 - tf.reduce_mean(tf.image.ssim_multiscale(self.logit_ph, self.output_ph, model_loss_info[1], MSSSIM_WEIGHTS2))) * (1 - model_loss_info[2])
		elif model_loss_info[0] == 22: # L2 + MS-SSIM
			self.model_loss = tf.reduce_sum(tf.reduce_mean(tf.square(tf.subtract(self.logit_ph, self.output_ph)), axis=0)) * model_loss_info[2]
			self.model_loss += (1 - tf.reduce_mean(tf.image.ssim_multiscale(self.logit_ph, self.output_ph, model_loss_info[1], MSSSIM_WEIGHTS2))) * (1 - model_loss_info[2])
		elif model_loss_info[0] == 100: # cross-entropy (for classification)
			#self.model_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logit_ph, labels=self.output_ph))
			self.model_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logit_ph, labels=self.output_ph))
			self.pred_ph = tf.nn.softmax(self.logit_ph)
		elif model_loss_info[0] == 101: # cross-entropy (for image segmentation)
			self.model_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.output_ph, logits=self.logit_ph)
			self.pred_ph = tf.nn.softmax(self.logit_ph)

		# setup loss
		if self.model_adv:
			# setup discriminator loss
			self.D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_real_logits_ph, labels=tf.ones_like(self.D_real_logits_ph)))  #labels=tf.ones([None, 1, 1, 1])))
			self.D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits_ph, labels=tf.zeros_like(self.D_fake_logits_ph)))
			self.D_loss = self.D_loss_real + self.D_loss_fake
			
			# finalize model loss
			self.G_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits_ph, labels=tf.ones_like(self.D_fake_logits_ph)))
			self.G_loss = self.model_loss + model_loss_info[3] * self.G_adv
			self.loss = self.G_loss

			self.t_vars = tf.trainable_variables()
			self.d_vars = [var for var in self.t_vars if var.name.startswith(self.model_dsc_name)]
			self.c_vars = list(set(self.t_vars) - set(self.d_vars))
			#self.c_vars = [var for var in self.t_vars if var.name.startswith(self.model_name)]

			tf.summary.scalar('loss', self.loss)
			tf.summary.scalar('dsc_loss_real', self.D_loss_real)
			tf.summary.scalar('dsc_loss_fake', self.D_loss_fake)
			tf.summary.scalar('dsc_loss', self.D_loss)
			tf.summary.scalar('gen_loss', self.G_loss)
		else:
			self.loss = self.model_loss
			tf.summary.scalar('loss', self.loss)

		# setup learning rate and decay
		self.global_step = tf.Variable(0, trainable=False)
		if decay_info[0] == 1:
			self.learning_rate = tf.train.exponential_decay(optimizer_info[1], self.global_step,
												   decay_info[1], decay_info[2], staircase=True)
		else:
			self.learning_rate = optimizer_info[1]
		
		# setup optimizer
		if self.model_adv:
			with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
				self.D_optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=optimizer_info[2], beta2=optimizer_info[3], epsilon=optimizer_info[4]).minimize(self.D_loss, var_list=self.d_vars, global_step=self.global_step)
				self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=optimizer_info[2], beta2=optimizer_info[3], epsilon=optimizer_info[4]).minimize(self.loss, var_list=self.c_vars, global_step=self.global_step)
		else:
			self.optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=optimizer_info[2], beta2=optimizer_info[3], epsilon=optimizer_info[4]).minimize(self.loss, global_step=self.global_step)

		# for tensorboard
		self.merge_summary = tf.summary.merge_all()
		
		# get default input and output extension
		self.input_ext = ''
		self.output_ext = ''
		if len(self.data_reader.test_list) > 0:
			input_info, output_info = self.data_reader.test_list[0]
			self.input_ext = os.path.splitext(input_info)[1][1:]
			self.output_ext = os.path.splitext(output_info)[1][1:]


	def sort_online(self, x_batch, y_batch, loss_batch, online_batch_size):
		loss_inds = np.argsort(loss_batch, axis=None)
		loss_inds = loss_inds[:online_batch_size-1:-1]  # descending, larger distance is bad
		#print(loss_batch)
		#print(loss_inds)
		x_batch_partial = x_batch[loss_inds]
		y_batch_partial = y_batch[loss_inds]
		loss_batch_partial = np.asarray(loss_batch[0])[loss_inds]
		#print(loss_batch_partial)
		return x_batch_partial, y_batch_partial, loss_batch_partial


	def train(self, epoch_count, batch_size, online_batch_size=0, save_rate=10, verbose=1, val_each_epoch=True, val_batch_size=0, test_save_output_batch=[], early_stop=False, output_summary_dir=''):
		if val_batch_size == 0:
			val_batch_size = batch_size
		if not output_summary_dir:
			output_summary_dir = self.output_dir
	
		# start session
		saver = tf.train.Saver()
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.9
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
		
			# load saved model if available
			ckpt = tf.train.get_checkpoint_state(self.output_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				if verbose >= 1:
					print('### checkpoint found -> model restored!')
	
			train_writer = tf.summary.FileWriter(self.output_dir, sess.graph)
			output_train_results = []
			output_val_results = []
			
			self.data_reader.begin_val(val_batch_size)
			val_error = np.ndarray(shape=(self.data_reader.val_batch_count * self.data_reader.val_batch_size), dtype=np.float32)

			for epoch_ind in range(epoch_count):
				if verbose >= 1:
					print('epoch - %d/%d' % (epoch_ind+1, epoch_count))

				self.data_reader.begin_train(batch_size)
				l_avg = 0
				for batch_ind in range(self.data_reader.train_batch_count):
					
					x_batch, y_batch = self.data_reader.next_train()
					if online_batch_size == 0:
						feed_dict = {self.input_ph: x_batch, self.output_ph: y_batch, self.training_phase_ph: 1, self.training_epoch_ph: epoch_ind}
						if self.model_adv:
							_, _, l, dl, al, lr, m_str, y_batch_pred = sess.run([self.optimizer, self.D_optimizer, self.loss, self.D_loss, self.G_adv, self.learning_rate, self.merge_summary, self.logit_ph], feed_dict=feed_dict)
						else:
							_, l, lr, m_str, y_batch_pred = sess.run([self.optimizer, self.loss, self.learning_rate, self.merge_summary, self.logit_ph], feed_dict=feed_dict)
					else:
						feed_dict = {self.input_ph: x_batch, self.output_ph: y_batch, self.training_phase_ph: 0, self.training_epoch_ph: epoch_ind}
						if self.model_adv:
							loss_batch = sess.run([self.model_loss_each], feed_dict=feed_dict)
						else:
							loss_batch = sess.run([self.model_loss_each], feed_dict=feed_dict)
						
						x_batch_partial, y_batch_partial, loss_batch_partial = self.sort_online(x_batch, y_batch, loss_batch, online_batch_size)
						feed_dict = {self.input_ph: x_batch_partial, self.output_ph: y_batch_partial, self.training_phase_ph: 1, self.training_epoch_ph: epoch_ind}
						if self.model_adv:
							_, _, l, dl, al, lr, m_str, y_batch_pred = sess.run([self.optimizer, self.D_optimizer, self.loss, self.D_loss, self.G_adv, self.learning_rate, self.merge_summary, self.logit_ph], feed_dict=feed_dict)
						else:
							_, l, lr, m_str, y_batch_pred = sess.run([self.optimizer, self.loss, self.learning_rate, self.merge_summary, self.logit_ph], feed_dict=feed_dict)
			
					l_avg += l
					if verbose == 2:
						print('[Training] [%d-%d]-[%d-%d] learning rate: %.5f, mini-batch loss: %.3f' % (epoch_ind+1, epoch_count, batch_ind+1, self.data_reader.train_batch_count, lr, l))
						if self.model_adv:
							print('dsc loss: %.3f, adv loss: %.3f' % (dl, al))
					
					if save_rate > 0 and (batch_ind % save_rate == 0 or batch_ind == self.data_reader.train_batch_count - 1):
						model_file = "model_%03d_%05d.ckpt" % (epoch_ind+1, batch_ind+1)
						model_path = os.path.join(self.output_dir, model_file)
						saver.save(sess, model_path)
						if verbose == 2:
							print('model saved: %s' % model_path)
					
					if batch_ind % 10 == 1:
						train_writer.add_summary(m_str, epoch_ind)

				l_avg = l_avg / self.data_reader.train_batch_count
				output_train_results.append(l_avg)
				if verbose >= 1:
					print('[Training] %d samples trained, loss: %f' % (self.data_reader.train_batch_count * batch_size, l_avg))

				if val_each_epoch:
					l_avg = 0
					self.data_reader.begin_val(val_batch_size)
					for batch_ind in range(self.data_reader.val_batch_count):

						x_batch, y_batch = self.data_reader.next_val()
						y_batch_loss, y_batch_pred = sess.run([self.loss, self.logit_ph], feed_dict={self.input_ph: x_batch, self.output_ph: y_batch, self.training_phase_ph: 0, self.training_epoch_ph: epoch_ind})
						l_avg += y_batch_loss
						if verbose == 2:
							print('[Validating] [%d-%d]-[%d-%d] val loss: %.3f' % (epoch_ind+1, epoch_count, batch_ind+1, self.data_reader.val_batch_count, y_batch_loss))
				
						sind = (batch_ind) * val_batch_size
						for n in range(val_batch_size):
							val_error[sind + n] = np.linalg.norm(y_batch[n] - y_batch_pred[n])

					l_avg = l_avg / self.data_reader.val_batch_count
					val_error_avg = np.mean(val_error)
					output_val_results.append([l_avg, val_error_avg])
					if len(output_val_results) > 2 and l_avg <= min(np.asarray(output_val_results)[:,0]):
						model_file = "model_%03d.ckpt" % (epoch_ind+1)
						model_path = os.path.join(self.output_dir, model_file)
						saver.save(sess, model_path)
						if verbose == 2:
							print('model saved: %s' % model_path)
					
					if verbose >= 1:
						print('[Validating] L2 distance: %.3f' % val_error_avg)

				if len(test_save_output_batch) > 0:
					x_batch, y_batch, _ = self.data_reader.get_test(test_save_output_batch)
					y_batch_pred = sess.run(self.logit_ph, feed_dict={self.input_ph: x_batch, self.training_phase_ph: 0, self.training_epoch_ph: epoch_ind})
						
					# save the output list to output format first
					for n, output_ind in enumerate(test_save_output_batch):
						output_gt_file = "output_%07d_gt.%s" % (output_ind, self.output_ext)
						output_pred_file = "output_%07d_%03d.%s" % (output_ind, epoch_ind + 1, self.output_ext)
						self.data_reader.__save_output__(y_batch[n], os.path.join(output_summary_dir, output_gt_file))
						self.data_reader.__save_output__(y_batch_pred[n], os.path.join(output_summary_dir, output_pred_file))

					# if output format is not image, save image as well
					if self.data_reader.output_type != 2 and self.data_reader.output_type != 10:
						for n, output_ind in enumerate(test_save_output_batch):
							output_gt_file = "output_%07d_gt.png" % (output_ind)
							output_pred_file = "output_%07d_%03d.png" % (output_ind, epoch_ind + 1)
							otype = self.data_reader.output_type
							self.data_reader.output_type = 2
							self.data_reader.__save_output__(y_batch[n], os.path.join(output_summary_dir, output_gt_file))
							self.data_reader.__save_output__(y_batch_pred[n], os.path.join(output_summary_dir, output_pred_file))
							self.data_reader.output_type = otype


			output_train_results_file = "output_train_summary.txt"
			with open(os.path.join(output_summary_dir, output_train_results_file), 'w') as output_fp:
				for ind, loss in enumerate(output_train_results):
					out_str = '%3d %10.4f' % (ind+1, loss)
					output_fp.write(out_str)
					output_fp.write('\n')
			if len(output_val_results) > 0:
				output_val_results_file = "output_val_summary.txt"
				with open(os.path.join(output_summary_dir, output_val_results_file), 'w') as output_fp:
					for ind, (loss, error) in enumerate(output_val_results):
						out_str = '%3d %10.4f %10.4f' % (ind+1, loss, error)
						output_fp.write(out_str)
						output_fp.write('\n')
						


	def test(self, test_batch_size, verbose=1, test_save_output=False, test_save_output_list=[], output_summary_dir=''):
		if not output_summary_dir:
			output_summary_dir = self.output_dir
	
		# start session
		saver = tf.train.Saver()
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.9
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			
			# load saved model if available
			ckpt = tf.train.get_checkpoint_state(self.output_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				if verbose >= 1:
					print('### checkpoint found -> model restored!')
		
			self.data_reader.begin_test(test_batch_size)
			
			y_error = np.ndarray(shape=(self.data_reader.test_batch_count * self.data_reader.test_batch_size), dtype=np.float32)
			if self.data_reader.output_type == 2 or self.data_reader.output_type == 3:
				y_rmse = np.ndarray(shape=(self.data_reader.test_batch_count * self.data_reader.test_batch_size), dtype=np.float32)
				y_ssim = np.ndarray(shape=(self.data_reader.test_batch_count * self.data_reader.test_batch_size), dtype=np.float32)
				y_psnr = np.ndarray(shape=(self.data_reader.test_batch_count * self.data_reader.test_batch_size), dtype=np.float32)
			elif self.data_reader.output_type == 10:
				y_label = np.ndarray(shape=(self.data_reader.test_batch_count * self.data_reader.test_batch_size), dtype=np.float32)
				y_pred  = np.ndarray(shape=(self.data_reader.test_batch_count * self.data_reader.test_batch_size), dtype=np.float32)
			
			for batch_ind in range(self.data_reader.test_batch_count):
				x_batch, y_batch = self.data_reader.next_test()
				test_loss, y_batch_pred = sess.run([self.loss, self.logit_ph], feed_dict={self.input_ph: x_batch, self.output_ph: y_batch, self.training_phase_ph: 0, self.training_epoch_ph: 0})
				if verbose == 2:
					print('[Testing] [%d-%d] test loss: %.3f' % (batch_ind+1, self.data_reader.test_batch_count, test_loss))
			
				ind1 = (batch_ind) * test_batch_size
				ind2 = (batch_ind+1) * test_batch_size
				for n in range(test_batch_size):
					y_error[ind1:ind2] = np.linalg.norm(y_batch[n] - y_batch_pred[n])
					if self.data_reader.output_type == 2 or self.data_reader.output_type == 3:
						y_rmse[ind1:ind2] = math.sqrt(compare_mse(y_batch[n], y_batch_pred[n]))
						y_ssim[ind1:ind2] = compare_ssim(y_batch[n], y_batch_pred[n], data_range=y_batch.max()-y_batch.min(), multichannel=True)
						y_psnr[ind1:ind2] = compare_psnr(y_batch[n], y_batch_pred[n], data_range=y_batch.max()-y_batch.min())
				
				#y_error[ind1:ind2] = np.linalg.norm(y_batch - y_batch_pred, axis=1)
				if self.data_reader.output_type == 10:
					y_label[ind1:ind2] = y_batch[:,0]
					y_pred[ind1:ind2]  = y_batch_pred[:,0]

			if verbose >= 1:
				l2 = np.mean(y_error)
				if self.data_reader.output_type == 2 or self.data_reader.output_type == 3:
					rmse = np.mean(y_rmse)
					ssim = np.mean(y_ssim)
					psnr = np.mean(y_psnr)
					print('[Testing] L2 error: %.3f, RMSE: %.3f, SSIM: %.3f, PSNR: %.3f' % (l2, rmse, ssim, psnr))
				elif self.data_reader.output_type == 10:
					rmse = math.sqrt(mean_squared_error(y_label, y_pred))
					mae = mean_absolute_error(y_label, y_pred)
					r2 = r2_score(y_label, y_pred)
					pearson, ppval = pearsonr(y_label, y_pred)
					spearman, spval = spearmanr(y_label, y_pred)
					mean = np.mean(y_pred)
					std = np.std(y_pred)
					print('[Testing] L2 error: %.3f, RMSE: %.3f, MAE: %.3f, R^2 score: %.3f, Pearson: %.3f, Spearman: %.3f, mean/std: %.3f/%.3f' % (l2, rmse, mae, r2, pearson, spearman, mean, std))
				else:
					print('[Testing] L2 error: %.3f' % (l2))

			if test_save_output:
				output_file = "output_test_pred.txt"
				with open(os.path.join(output_summary_dir, output_file), 'w') as output_fp:
					for out_ind in range(y_error.shape[0]):
						if self.data_reader.output_type == 2 or self.data_reader.output_type == 3:
							out_str = '%10.4f %10.4f %10.4f' % (y_rmse[out_ind], y_ssim[out_ind], y_psnr[out_ind])
						elif self.data_reader.output_type == 10:
							out_str = '%10.4f %10.4f %10.4f' % (y_label[out_ind], y_pred[out_ind], y_error[out_ind])
						output_fp.write(out_str)
						output_fp.write('\n')

			if len(test_save_output_list) > 0:
				x_output, y_output, info_output = self.data_reader.get_test(test_save_output_list)
				y_output_pred = np.zeros_like(y_output)
				
				test_batch_count = int(len(test_save_output_list) / test_batch_size)
				test_remain_count = int(len(test_save_output_list) % test_batch_size)
				for batch_ind in range(test_batch_count):
					ind1 = batch_ind * test_batch_size
					ind2 = (batch_ind + 1) * test_batch_size
					x_batch = x_output[ind1:ind2,...]
					y_batch_pred = sess.run(self.logit_ph, feed_dict={self.input_ph: x_batch, self.training_phase_ph: 0, self.training_epoch_ph: 0})
					y_output_pred[ind1:ind2,...] = y_batch_pred
					if verbose == 2:
						print('[Testing output] [%d-%d]' % (batch_ind+1, test_batch_count))
				
				if test_remain_count > 0:
					x_batch = x_output[test_batch_count*test_batch_size:len(test_save_output_list),...]
					y_batch_pred = sess.run(self.logit_ph, feed_dict={self.input_ph: x_batch, self.training_phase_ph: 0, self.training_epoch_ph: 0})
					y_output_pred[test_batch_count*test_batch_size:len(test_save_output_list),...] = y_batch_pred

				# save the output list to output format first
				for n, output_ind in enumerate(test_save_output_list):
					input_file = "input_%07d.%s" % (output_ind, self.input_ext)
					output_gt_file = "output_%07d_gt.%s" % (output_ind, self.output_ext)
					output_pred_file = "output_%07d_pred.%s" % (output_ind, self.output_ext)
					self.data_reader.__save_input__(x_output[n], os.path.join(output_summary_dir, input_file))
					self.data_reader.__save_output__(y_output[n], os.path.join(output_summary_dir, output_gt_file))
					self.data_reader.__save_output__(y_output_pred[n], os.path.join(output_summary_dir, output_pred_file))

				# if output format is not image, save image as well
				if self.data_reader.output_type != 2 and self.data_reader.output_type != 10:
					for n, output_ind in enumerate(test_save_output_list):
						input_file = "input_%07d.png" % (output_ind)
						output_gt_file = "output_%07d_gt.png" % (output_ind)
						output_pred_file = "output_%07d_pred.png" % (output_ind)
						itype = self.data_reader.input_type
						otype = self.data_reader.output_type
						self.data_reader.input_type = 2
						self.data_reader.__save_input__(x_output[n], os.path.join(output_summary_dir, input_file))
						self.data_reader.input_type = itype
						self.data_reader.output_type = 2
						self.data_reader.__save_output__(y_output[n], os.path.join(output_summary_dir, output_gt_file))
						self.data_reader.__save_output__(y_output_pred[n], os.path.join(output_summary_dir, output_pred_file))
						self.data_reader.output_type = otype
				

	def train_custom(self, batch_size, shuffle, run_custom_batch, custom_vars=[]):
		# start session
		saver = tf.train.Saver()
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.9
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			
			# load saved model if available
			ckpt = tf.train.get_checkpoint_state(self.output_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print("### checkpoint found -> model restored!")

			self.data_reader.begin_train(batch_size, shuffle=shuffle)
			for batch_ind in range(self.data_reader.train_batch_count):
				x_batch, y_batch = self.data_reader.next_train()
				run_custom_batch(self, sess, saver, batch_ind, batch_size, x_batch, y_batch, custom_vars)


	def val_custom(self, batch_size, run_custom_batch, custom_vars=[]):
		# start session
		saver = tf.train.Saver()
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.9
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			
			# load saved model if available
			ckpt = tf.train.get_checkpoint_state(self.output_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print("### checkpoint found -> model restored!")
			
			self.data_reader.begin_val(batch_size)
			for batch_ind in range(self.data_reader.val_batch_count):
				x_batch, y_batch = self.data_reader.next_val()
				run_custom_batch(self, sess, saver, batch_ind, batch_size, x_batch, y_batch, custom_vars)


	def test_custom(self, batch_size, run_custom_batch, custom_vars=[]):
		# start session
		saver = tf.train.Saver()
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.9
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			
			# load saved model if available
			ckpt = tf.train.get_checkpoint_state(self.output_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
				print("### checkpoint found -> model restored!")
			
			self.data_reader.begin_test(batch_size)
			for batch_ind in range(self.data_reader.test_batch_count):
				x_batch, y_batch = self.data_reader.next_test()
				run_custom_batch(self, sess, saver, batch_ind, batch_size, x_batch, y_batch, custom_vars)


	def begin_sess_custom(self):
		# start session
		saver = tf.train.Saver()
		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.9
		sess = tf.Session(config=config)
		sess.run(tf.global_variables_initializer())
		
		# load saved model if available
		ckpt = tf.train.get_checkpoint_state(self.output_dir)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("### checkpoint found -> model restored!")
		
		return sess, saver


	def test_sess_custom(self, sess, saver, batch_size, run_custom_batch, custom_vars=[]):
		self.data_reader.begin_test(batch_size)
		for batch_ind in range(self.data_reader.test_batch_count):
			x_batch, y_batch = self.data_reader.next_test()
			run_custom_batch(self, sess, saver, batch_ind, batch_size, x_batch, y_batch, custom_vars)

