################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Evaluation metric utility functions
################################################################################


from __future__ import division

import warnings
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.utils import check_consistent_length
from sklearn.utils import column_or_1d, check_array
#from sklearn.utils.multiclass import type_of_target
#from sklearn.utils.fixes import isclose
#from sklearn.utils.fixes import bincount
#from sklearn.utils.fixes import array_equal
#from sklearn.utils.stats import rankdata
#from sklearn.utils.sparsefuncs import count_nonzero

from sklearn.metrics.base import _average_binary_score


def _parse_version(version_string):
	version = []
	for x in version_string.split('.'):
		try:
			version.append(int(x))
		except ValueError:
			version.append(x)
	return tuple(version)


np_version = _parse_version(np.__version__)

if np_version < (1, 8, 1):
	def array_equal(a1, a2):
		# copy-paste from numpy 1.8.1
		try:
			a1, a2 = np.asarray(a1), np.asarray(a2)
		except:
			return False
		if a1.shape != a2.shape:
			return False
		return bool(np.asarray(a1 == a2).all())
else:
	from numpy import array_equal


def auc(x, y, reorder=False):
	check_consistent_length(x, y)
	x = column_or_1d(x)
	y = column_or_1d(y)

	if x.shape[0] < 2:
		raise ValueError('At least 2 points are needed to compute'' area under curve, but x.shape = %s' % x.shape)

	direction = 1
	if reorder:
		order = np.lexsort((y, x))
		x, y = x[order], y[order]
	else:
		dx = np.diff(x)
		if np.any(dx < 0):
			if np.all(dx <= 0):
				direction = -1
			else:
				raise ValueError("Reordering is not turned on, and ""the x array is not increasing: %s" % x)

	area = direction * np.trapz(y, x)
	if isinstance(area, np.memmap):
		area = area.dtype.type(area)

	return area



def average_precision_score_hj(y_true, y_score, average="macro", sample_weight=None):
	def _binary_average_precision(y_true, y_score, sample_weight=None):
		precision, recall, thresholds = precision_recall_curve_hj(y_true, y_score, sample_weight=sample_weight)
		return auc(recall, precision)

	return _average_binary_score(_binary_average_precision, y_true, y_score, average, sample_weight=sample_weight)


def precision_recall_curve_hj(y_true, probas_pred, pos_label=None, sample_weight=None):

	fps, tps, thresholds = _binary_clf_curve(y_true, probas_pred, pos_label=pos_label, sample_weight=sample_weight)

	precision = tps / (tps + fps)
	#recall = tps / tps[-1]
	recall = np.ones(tps.size) if tps[-1] == 0 else tps / tps[-1]

	# stop when full recall attained
	# and reverse the outputs so recall is decreasing
	last_ind = tps.searchsorted(tps[-1])
	sl = slice(last_ind, None, -1)
	return np.r_[precision[sl], 1], np.r_[recall[sl], 0], thresholds[sl]


def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
	check_consistent_length(y_true, y_score)
	y_true = column_or_1d(y_true)
	y_score = column_or_1d(y_score)
	if sample_weight is not None:
		sample_weight = column_or_1d(sample_weight)

	classes = np.unique(y_true)
	if (pos_label is None and not (array_equal(classes, [0, 1]) or array_equal(classes, [-1, 1]) or array_equal(classes, [0]) or array_equal(classes, [-1]) or array_equal(classes, [1]))):
		raise ValueError("Data is not binary and pos_label is not specified")
	elif pos_label is None:
		pos_label = 1.

	y_true = (y_true == pos_label)

	desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
	y_score = y_score[desc_score_indices]
	y_true = y_true[desc_score_indices]
	if sample_weight is not None:
		weight = sample_weight[desc_score_indices]
	else:
		weight = 1.

	distinct_value_indices = np.where(np.logical_not(np.isclose(np.diff(y_score), 0)))[0]
	threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

	tps = (y_true * weight).cumsum()[threshold_idxs]
	if sample_weight is not None:
		fps = weight.cumsum()[threshold_idxs] - tps
	else:
		fps = 1 + threshold_idxs - tps

	return fps, tps, y_score[threshold_idxs]

