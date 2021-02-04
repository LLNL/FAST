################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Tensorflow utility functions
################################################################################


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf



def bias_var(shape, name):
	initial = tf.constant(0.0, shape=shape)
	return tf.get_variable(name, initializer=initial)

def weight_var(shape, stddev, name):
	initial = tf.random_normal(shape, stddev=stddev)
	return tf.get_variable(name, initializer=initial)

def weight_var_selu(shape, name): # no issue with vanishing gradients, faster and better than relu
	stddev = np.sqrt(1.0 / (shape[0] * shape[1] * shape[2] * shape[4]))
	return weight_var(shape, stddev=stddev, name=name)

def lrelu(x, leak=0.2, name="lrelu"):
	with tf.variable_scope(name):
		f1 = 0.5 * (1 + leak)
		f2 = 0.5 * (1 - leak)
		return f1 * x + f2 * tf.abs(x)

def bn(x,is_training,name):
	return tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None, is_training=is_training, reuse=None, trainable=True, scope=name)

def conv3d(x, W, filter_size=5, strides=(1,2,2,2,1)):
	return tf.nn.conv3d(x, W, strides=strides, padding="SAME")

def max_pool_2x2x2(x):
	return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding="SAME")

def max_pool_1x2x2(x):
	return tf.nn.max_pool3d(x, ksize=[1, 1, 2, 2, 1], strides=[1, 1, 2, 2, 1], padding="SAME")

def avg_pool_1x2x2(x):
	return tf.layers.average_pooling3d(x, pool_size=[2, 2, 2], strides=[1, 2, 2], padding="SAME")

def avg_pool_2x2x2(x):
	return tf.layers.average_pooling3d(x, pool_size=[2, 2, 2], strides=[2, 2, 2], padding="SAME")

