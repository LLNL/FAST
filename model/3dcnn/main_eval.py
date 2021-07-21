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
sys.stdout.flush()
sys.path.insert(0, "../common")
import argparse
import random
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import *
from scipy.stats import *
from model import Model_3DCNN, strip_prefix_if_present
from data_reader import Dataset_MLHDF
from img_util import GaussianFilter, Voxelizer3D
from file_util import *



# program arguments
parser = argparse.ArgumentParser()
parser.add_argument("--device-name", default="cuda:0", help="use cpu or cuda:0, cuda:1 ...")
parser.add_argument("--data-dir", default="/home/kim63/data", help="dataset directory")
parser.add_argument("--dataset-type", type=float, default=1, help="1: ml-hdf, 2: ml-hdf v2")
parser.add_argument("--csv-fn", default="", help="csv file path")
parser.add_argument("--mlhdf-fn", default="pdbbind2019_crystal_core_ml.hdf", help="ml-hdf path")
parser.add_argument("--model-path", default="/home/kim63/data/pdbbind2019_crystal_refined_model_20201216.pth", help="model checkpoint file path")
parser.add_argument("--complex-type", type=int, default=1, help="1: crystal, 2: docking")
parser.add_argument("--rmsd-threshold", type=float, default=2, help="rmsd cut-off threshold in case of docking data and/or --rmsd-weight is true")
parser.add_argument("--batch-size", type=int, default=50, help="mini-batch size")
parser.add_argument("--multi-gpus", default=False, action="store_true", help="whether to use multi-gpus")
parser.add_argument("--save-pred", default=True, action="store_true", help="whether to save prediction results in csv")
parser.add_argument("--save-feat", default=True, action="store_true", help="whether to save fully connected features in npy")
args = parser.parse_args()


# set CUDA for PyTorch
use_cuda = torch.cuda.is_available()
cuda_count = torch.cuda.device_count()
if use_cuda:
	device = torch.device(args.device_name)
	torch.cuda.set_device(int(args.device_name.split(':')[1]))
else:
	device = torch.device("cpu")
print(use_cuda, cuda_count, device)




def eval():

	# load dataset
	if args.complex_type == 1:
		is_crystal = True
	else:
		is_crystal = False
	dataset = Dataset_MLHDF(os.path.join(args.data_dir, args.mlhdf_fn), args.dataset_type, os.path.join(args.data_dir, args.csv_fn), is_crystal=is_crystal, rmsd_weight=False, rmsd_thres=args.rmsd_threshold)

	# check multi-gpus
	num_workers = 0
	if args.multi_gpus and cuda_count > 1:
		num_workers = cuda_count

	# initialize data loader
	batch_size = args.batch_size
	batch_count = len(dataset) // batch_size
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=None)

	# define voxelizer, gaussian_filter
	voxelizer = Voxelizer3D(use_cuda=use_cuda, verbose=0)
	gaussian_filter = GaussianFilter(dim=3, channels=19, kernel_size=5, sigma=1, use_cuda=use_cuda)

	# define model
	model = Model_3DCNN(use_cuda=use_cuda, verbose=0)
	#if use_cuda:
	#	model = model.cuda()
	if args.multi_gpus and cuda_count > 1:
		model = nn.DataParallel(model)
	model.to(device)

	if isinstance(model, (DistributedDataParallel, DataParallel)):
		model_to_save = model.module
	else:
		model_to_save = model

	# load model
	if not valid_file(args.model_path):
		print("checkpoint not found! %s" % args.model_path)
		return
	checkpoint = torch.load(args.model_path, map_location=device)
	#checkpoint = torch.load(args.model_path)
	model_state_dict = checkpoint.pop("model_state_dict")
	strip_prefix_if_present(model_state_dict, "module.")
	model_to_save.load_state_dict(model_state_dict, strict=False)
	output_dir = os.path.dirname(args.model_path)

	vol_batch = torch.zeros((batch_size,19,48,48,48)).float().to(device)
	ytrue_arr = np.zeros((len(dataset),), dtype=np.float32)
	ypred_arr = np.zeros((len(dataset),), dtype=np.float32)
	zfeat_arr = np.zeros((len(dataset), 100), dtype=np.float32)
	pred_list = []

	model.eval()
	with torch.no_grad():
		for bind, batch in enumerate(dataloader):
		
			# transfer to GPU
			x_batch_cpu, y_batch_cpu = batch
			x_batch, y_batch = x_batch_cpu.to(device), y_batch_cpu.to(device)
			
			# voxelize into 3d volume
			bsize = x_batch.shape[0]
			for i in range(bsize):
				xyz, feat = x_batch[i,:,:3], x_batch[i,:,3:]
				vol_batch[i,:,:,:,:] = voxelizer(xyz, feat)
			vol_batch = gaussian_filter(vol_batch)
			
			# forward training
			ypred_batch, zfeat_batch = model(vol_batch[:x_batch.shape[0]])

			ytrue = y_batch_cpu.float().data.numpy()[:,0]
			ypred = ypred_batch.cpu().float().data.numpy()[:,0]
			zfeat = zfeat_batch.cpu().float().data.numpy()
			ytrue_arr[bind*batch_size:bind*batch_size+bsize] = ytrue
			ypred_arr[bind*batch_size:bind*batch_size+bsize] = ypred
			zfeat_arr[bind*batch_size:bind*batch_size+bsize] = zfeat

			if args.save_pred:
				for i in range(bsize):
					pred_list.append([bind + i, ytrue[i], ypred[i]])

			print("[%d/%d] evaluating" % (bind+1, batch_count))
			#ytrue_str = np.array_repr(ytrue).replace('\n', '')
			#ypred_str = np.array_repr(ypred).replace('\n', '')
			#print(ytrue_str)
			#print(ypred_str)

	rmse = math.sqrt(mean_squared_error(ytrue_arr, ypred_arr))
	mae = mean_absolute_error(ytrue_arr, ypred_arr)
	r2 = r2_score(ytrue_arr, ypred_arr)
	pearson, ppval = pearsonr(ytrue_arr, ypred_arr)
	spearman, spval = spearmanr(ytrue_arr, ypred_arr)
	mean = np.mean(ypred_arr)
	std = np.std(ypred_arr)
	print("Evaluation Summary:")
	print("RMSE: %.3f, MAE: %.3f, R^2 score: %.3f, Pearson: %.3f, Spearman: %.3f, mean/std: %.3f/%.3f" % (rmse, mae, r2, pearson, spearman, mean, std))

	if args.save_pred:
		csv_fpath = "%s_%s_pred.csv" % (args.model_path[:-4], args.mlhdf_fn[:-4])
		df = pd.DataFrame(pred_list, columns=["cid", "label", "pred"])
		df.to_csv(csv_fpath, index=False)
		
	if args.save_feat:
		npy_fpath = "%s_%s_feat.npy" % (args.model_path[:-4], args.mlhdf_fn[:-4])
		np.save(npy_fpath, zfeat_arr)


def main():
	eval()


if __name__ == "__main__":
	main()


