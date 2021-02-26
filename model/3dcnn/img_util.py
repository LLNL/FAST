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



# previous implementation (get_3D_all2) for sanity check (with (z,y,x,c) to (c,z,y,x) conversion)
def voxelize_3d(xyz, feat, vol_dim=[19,48,48,48], relative_size=True, size_angstrom=48, atom_radii=None, atom_radius=1, sigma=0):

	# get 3d bounding box
	xmin = min(xyz[:, 0])
	ymin = min(xyz[:, 1])
	zmin = min(xyz[:, 2])
	xmax = max(xyz[:, 0])
	ymax = max(xyz[:, 1])
	zmax = max(xyz[:, 2])
	
	# initialize volume
	vol_data = np.zeros((vol_dim[0], vol_dim[1], vol_dim[2], vol_dim[3]), dtype=np.float32)

	if relative_size:
		# voxel size (assum voxel size is the same in all axis
		vox_size = float(zmax - zmin) / float(vol_dim[1])
	else:
		vox_size = float(size_angstrom) / float(vol_dim[1])
		xmid = (xmin + xmax) / 2.0
		ymid = (ymin + ymax) / 2.0
		zmid = (zmin + zmax) / 2.0
		xmin = xmid - (size_angstrom / 2)
		ymin = ymid - (size_angstrom / 2)
		zmin = zmid - (size_angstrom / 2)
		xmax = xmid + (size_angstrom / 2)
		ymax = ymid + (size_angstrom / 2)
		zmax = zmid + (size_angstrom / 2)
		vox_size2 = float(size_angstrom) / float(vol_dim[1])
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

		cx = (x - xmin) / (xmax - xmin) * (vol_dim[3] - 1)
		cy = (y - ymin) / (ymax - ymin) * (vol_dim[2] - 1)
		cz = (z - zmin) / (zmax - zmin) * (vol_dim[1] - 1)

		vx_from = max(0, int(cx - atom_radius))
		vx_to = min(vol_dim[3] - 1, int(cx + atom_radius))
		vy_from = max(0, int(cy - atom_radius))
		vy_to = min(vol_dim[2] - 1, int(cy + atom_radius))
		vz_from = max(0, int(cz - atom_radius))
		vz_to = min(vol_dim[1] - 1, int(cz + atom_radius))
		
		for vz in range(vz_from, vz_to + 1):
			for vy in range(vy_from, vy_to + 1):
				for vx in range(vx_from, vx_to + 1):
						vol_data[:, vz, vy, vx] += feat[ind, :]

	# gaussian filter
	if sigma > 0:
		for i in range(vol_data.shape[0]):
			vol_data[i,:,:,:] = sp.ndimage.filters.gaussian_filter(vol_data[i,:,:,:], sigma=sigma, truncate=2)

	return vol_data



# apply to individual volume (not a batch)
class Voxelizer3D(nn.Module):
	def __init__(self, feat_dim=19, vol_dim=48, ang_size=48, relative_size=True, atom_radius=1, use_cuda=True, verbose=0):
		super(Voxelizer3D, self).__init__()
		
		self.feat_dim = feat_dim
		self.vol_dim = vol_dim
		self.ang_size = ang_size
		self.relative_size = relative_size
		self.atom_radius = atom_radius
		self.use_cuda = use_cuda
		self.verbose = verbose

	def forward(self, xyz, feat, atom_radii=None):
		# filter out zero padded rows
		mask = (xyz[:,0] != 0) & (xyz[:,1] != 0) & (xyz[:,2] != 0)
		xyz = xyz[mask]
		feat = feat[mask]
	
		# get 3d bounding box
		xmin, ymin, zmin = min(xyz[:,0]), min(xyz[:,1]), min(xyz[:,2])
		xmax, ymax, zmax = max(xyz[:,0]), max(xyz[:,1]), max(xyz[:,2])
		if self.relative_size:
			# voxel size (assuming voxel size is the same in all axis)
			vox_size = float(zmax - zmin) / float(self.vol_dim)
		else:
			vox_size = float(self.ang_size) / float(self.vol_dim)
			xmid, ymid, zmid = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0, (zmin + zmax) / 2.0
			xmin, ymin, zmin = xmid - (self.ang_size / 2), ymid - (self.ang_size / 2), zmid - (self.ang_size / 2)
			xmax, ymax, zmax = xmid + (self.ang_size / 2), ymid + (self.ang_size / 2), zmid + (self.ang_size / 2)

		# initialize vol data
		if self.use_cuda:
			vol_data = torch.cuda.FloatTensor(self.vol_dim, self.vol_dim, self.vol_dim, self.feat_dim).fill_(0)
		else:
			vol_data = torch.zeros((self.vol_dim, self.vol_dim, self.vol_dim, self.feat_dim)).float()

		# assign each atom to voxels
		for ind in range(xyz.shape[0]):
			x, y, z = xyz[ind, 0], xyz[ind, 1], xyz[ind, 2]
			if x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax:
				continue

			# compute van der Waals radius and atomic density, use 1 if not available
			if not atom_radii is None:
				vdw_radius = atom_radii[ind]
				atom_radius = 1 + vdw_radius * vox_size
			else:
				atom_radius = self.atom_radius

			cx = int((x-xmin) / (xmax-xmin) * (self.vol_dim-1))
			cy = int((y-ymin) / (ymax-ymin) * (self.vol_dim-1))
			cz = int((z-zmin) / (zmax-zmin) * (self.vol_dim-1))
			vx_from = max(0, int(cx-atom_radius))
			vx_to = min(self.vol_dim-1, int(cx+atom_radius))
			vy_from = max(0, int(cy-atom_radius))
			vy_to = min(self.vol_dim-1, int(cy+atom_radius))
			vz_from = max(0, int(cz-atom_radius))
			vz_to = min(self.vol_dim-1, int(cz+atom_radius))
			
			vol_feat = feat[ind,:].repeat(vz_to-vz_from+1, vy_to-vy_from+1, vx_to-vx_from+1, 1)
			vol_data[vz_from:vz_to+1, vy_from:vy_to+1, vx_from:vx_to+1, :] += vol_feat
			
			# below code doesn't work as dimension can be different if atom is located near boundary
			#vol_feat = feat[ind,:].repeat(atom_radius*2+1, atom_radius*2+1, atom_radius*2+1, 1)
			#vol_data[cz-atom_radius:cz+atom_radius+1, cy-atom_radius:cy+atom_radius+1, cx-atom_radius:cx+atom_radius+1, :] += vol_feat
				
		vol_data = vol_data.permute(3,0,1,2) #-> doesn't need as we already initialized 19x48x48x48
		return vol_data



# apply to volume batch
# kernerl_size=11, sigma=1 is equivalent to
# sp.ndimage.filters.gaussian_filter(..., sigma=1, truncate=4.5, mode='constant')
class GaussianFilter(nn.Module):
	def __init__(self, dim=2, channels=3, kernel_size=11, sigma=1, use_cuda=True):
		super(GaussianFilter, self).__init__()
		
		self.use_cuda = use_cuda
		if isinstance(kernel_size, numbers.Number):
			self.kernel_size = [kernel_size] * dim
		if isinstance(sigma, numbers.Number):
			self.sigma = [sigma] * dim
			
		self.padding = kernel_size // 2

		# Gaussian kernel is the product of the gaussian function of each dimension.
		kernel = 1
		meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in self.kernel_size])
		for size, std, mgrid in zip(self.kernel_size, self.sigma, meshgrids):
			mean = (size - 1) / 2
			kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

		# make sure sum of values in gaussian kernel equals 1.
		kernel = kernel / torch.sum(kernel)

		# reshape to depthwise convolutional weight
		kernel = kernel.view(1, 1, *kernel.size()) #-> doesn't need to add one more axis
		#kernel = kernel.view(1, *kernel.size())
		kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))
		if self.use_cuda:
			kernel = kernel.cuda()

		self.register_buffer('weight', kernel)
		self.groups = channels
		if dim == 1:
			self.conv = nn.functional.conv1d
		elif dim == 2:
			self.conv = nn.functional.conv2d
		elif dim == 3:
			self.conv = nn.functional.conv3d

	def forward(self, input):
		return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)
		
