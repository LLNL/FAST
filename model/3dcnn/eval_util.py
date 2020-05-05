################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Evaluation util functions
################################################################################


from __future__ import absolute_import
from __future__ import print_function

import os
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.metrics import roc_curve, auc, accuracy_score #, average_precision_score
from metric_util import *


def generate_results(y_test, y_score):
	fpr, tpr, _ = roc_curve(y_test, y_score)
	roc_auc = auc(fpr, tpr)
	plt.figure()
	plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.05])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic curve')
	plt.show()
	plt.draw()
	plt.savefig("result_roc.png")
	print('AUC: %f' % roc_auc)


def evaluate_model(y_pred, y_test, y_test_list):
	
	y_pred_class = np.argmax(y_pred, axis=1).tolist()
	y_pred_score = np.max(y_pred, axis=1).tolist()
	
	# accuracy
	y_test_class = np.asarray(y_test_list)
	acc = accuracy_score(y_test_class, y_pred_class)
	
	# average precision
	AP = average_precision_score_hj(y_test.astype('float'), y_pred.astype('float'))
	
	return acc, AP


def evaluate_model_groups(y_pred, y_test, y_test_list, numgroups):
	
	y_pred_class = np.argmax(y_pred, axis=1).tolist()
	y_pred_score = np.max(y_pred, axis=1).tolist()
	
	# make per-group prediction
	y_pred_group_class = []
	y_pred_group_list = []
	for n in range(0, len(y_pred_class), numgroups):
		y_group = y_pred_class[n:n+numgroups]
		most_common,num_most_common = Counter(y_group).most_common(1)[0]
		y_pred_group_class.append(most_common)
		
		y_pred_g = np.average(y_pred[n:n+numgroups], axis=0)
		y_pred_group_list.append(y_pred_g)
	
	# per-group accuracy
	y_test_group_class = np.asarray(y_test_list[::numgroups])
	acc_group = accuracy_score(y_test_group_class, y_pred_group_class)
	
	# per-group average precision
	y_group_test = y_test[::numgroups].astype('float')
	y_pred_group = np.asarray(y_pred_group_list).astype('float')
	AP_group = average_precision_score_hj(y_group_test, y_pred_group)
	
	return acc_group, AP_group

