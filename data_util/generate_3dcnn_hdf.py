################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Generate 3D representation in hdf5 format from input hdf5 (e.g., data/core_test.hdf) for 3D-CNN model training
################################################################################


import os
import sys
import shutil
import argparse
import csv
import h5py
import numpy as np
import scipy as sp
import scipy.ndimage

#from file_util import *



parser = argparse.ArgumentParser()
parser.add_argument("--main-dir", default="../data", help="main dataset directory")
parser.add_argument("--use-external", default=True, help="whether external test file is used or not")
parser.add_argument("--input-file", default="", help="input test HDF filename")
parser.add_argument("--output-file", default="", help="output test HDF filename")
args = parser.parse_args()



# do not change unless the hdf structure is changed
g_csv_header = ['ligand_id', 'file_prefix', 'label', 'train_test_split', 'atom_count', 'xsize', 'ysize', 'zsize', 'p_atom_count1', 'p_atom_count2', 'p_xsize', 'p_ysize', 'p_zsize']

g_feat_tool_list = ['pybel', 'rdkit']
g_feat_tool_ind = 0

g_feat_type_list = ['raw', 'processed']
g_feat_type_ind = 1

g_feat_pdbbind_type_list = ['crystal', 'docking'] # for display
g_feat_pdbbind_type_list2 = ['pdbbind', 'docking'] # for reading hdf
g_feat_pdbbind_type_ind = 1

g_feat_data_str = 'data'


g_main_dir = "../data"

g_target_dataset = "pdbbind2016"
g_target_trainval_type = "refined"
g_external_test = True # default is False
if g_external_test:
    g_feat_suffix = "%s_%s_%s" % (g_feat_tool_list[g_feat_tool_ind], g_feat_type_list[g_feat_type_ind], g_feat_pdbbind_type_list[g_feat_pdbbind_type_ind])
else:
    g_feat_suffix = "%s_%s_%s_%s_%s" % (g_target_dataset, g_target_trainval_type, g_feat_tool_list[g_feat_tool_ind], g_feat_type_list[g_feat_type_ind], g_feat_pdbbind_type_list[g_feat_pdbbind_type_ind])

g_3D_relative_size = False
g_3D_size_angstrom = 48 # valid only when g_3D_relative_size = False
g_3D_size_dim = 48 # 48
g_3D_atom_radius = 1
g_3D_atom_radii = False
g_3D_sigma = 1
if g_feat_tool_ind == 0:
    g_3D_dim = [g_3D_size_dim, g_3D_size_dim, g_3D_size_dim, 19]
else:
    g_3D_dim = [g_3D_size_dim, g_3D_size_dim, g_3D_size_dim, 75]

size_angstrom = g_3D_size_angstrom
if g_3D_relative_size:
    size_angstrom = 0

if g_3D_atom_radii:
    g_3D_suffix = "%d_%d_radii_sigma%d_rot0" % (size_angstrom, g_3D_size_dim, g_3D_sigma)
else:
    g_3D_suffix = "%d_%d_radius%d_sigma%d_rot0" % (size_angstrom, g_3D_size_dim, g_3D_atom_radius, g_3D_sigma)

if g_external_test:
    g_input_test_hd_file = "core_test.hdf"
else:
    g_input_train_hd_file = "%s_%s_train.hdf" % (g_target_dataset, g_target_trainval_type)
    g_input_val_hd_file = "%s_%s_val.hdf" % (g_target_dataset, g_target_trainval_type)
    g_input_test_hd_file = "%s_core_test.hdf" % (g_target_dataset)

g_output_hd_compress = True
g_output_train_hd_file = "%s_%s_train.hdf" % (g_feat_suffix, g_3D_suffix)
g_output_val_hd_file = "%s_%s_val.hdf" % (g_feat_suffix, g_3D_suffix)
g_output_test_hd_file = "%s_%s_test.hdf" % (g_feat_suffix, g_3D_suffix)
g_output_csv = "%s_%s_info.csv" % (g_feat_suffix, g_3D_suffix)


# for argument setting
g_main_dir = args.main_dir
if args.use_external:
    g_input_test_hd_file = args.input_file
    g_output_test_hd_file = args.output_file
    g_output_csv = args.output_file[:-4] + ".csv"



def rotate_3D(input_data):
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval], [0, 1, 0], [-sinval, 0, cosval]])
    #rotated_data = np.zeros(input_data.shape, dtype=np.float32)
    rotated_data = np.dot(input_data, rotation_matrix)
    return rotated_data


def get_3D_bound(xyz_array):
    xmin = min(xyz_array[:, 0])
    ymin = min(xyz_array[:, 1])
    zmin = min(xyz_array[:, 2])
    xmax = max(xyz_array[:, 0])
    ymax = max(xyz_array[:, 1])
    zmax = max(xyz_array[:, 2])
    return xmin, ymin, zmin, xmax, ymax, zmax


def get_3D_all(xyz, feat, vol_dim, xmin, ymin, zmin, xmax, ymax, zmax, atom_radius=1, atomtype_ind=-1, sigma=0):

    # initialize volume
    vol_data = np.zeros((vol_dim[0], vol_dim[1], vol_dim[2], vol_dim[3]), dtype=np.float32)
    vol_tag = np.zeros((vol_dim[0], vol_dim[1], vol_dim[2]), dtype=np.int32)

    # voxel size (assum voxel size is the same in all axis
    vox_size = (zmax - zmin) / vol_dim[0]

    # assign xyz (only center)
    for ind in range(xyz.shape[0]):
        x = xyz[ind, 0]
        y = xyz[ind, 1]
        z = xyz[ind, 2]
        if x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax:
            continue

        cx = (x - xmin) / (xmax - xmin) * (vol_dim[2] - 1)
        cy = (y - ymin) / (ymax - ymin) * (vol_dim[1] - 1)
        cz = (z - zmin) / (zmax - zmin) * (vol_dim[0] - 1)

        vol_tag[int(cz), int(cy), int(cx)] += 1
        if vol_tag[int(cz), int(cy), int(cx)] == 1:
            vol_data[int(cz), int(cy), int(cx), :] = feat[ind, :]

    # assign xyz
    for ind in range(xyz.shape[0]):
        x = xyz[ind, 0]
        y = xyz[ind, 1]
        z = xyz[ind, 2]
        if x < xmin or x > xmax or y < ymin or y > ymax or z < zmin or z > zmax:
            continue

        # compute van der Waals radius and atomic density, use 1 if not available
        if atomtype_ind >= 0:
            vdw_radius = g_atom_vdw_ligand[feat[ind, atomtype_ind]]
            atom_radius = 1 + vdw_radius * vox_size

        # setup atom ranges
        cx = (x - xmin) / (xmax - xmin) * (vol_dim[2] - 1)
        cy = (y - ymin) / (ymax - ymin) * (vol_dim[1] - 1)
        cz = (z - zmin) / (zmax - zmin) * (vol_dim[0] - 1)

        vx_from = max(0, int(cx - atom_radius))
        vx_to = min(vol_dim[2] - 1, int(cx + atom_radius))
        vy_from = max(0, int(cy - atom_radius))
        vy_to = min(vol_dim[1] - 1, int(cy + atom_radius))
        vz_from = max(0, int(cz - atom_radius))
        vz_to = min(vol_dim[0] - 1, int(cz + atom_radius))

        # uniform density
        for vz in range(vz_from, vz_to + 1):
            for vy in range(vy_from, vy_to + 1):
                for vx in range(vx_from, vx_to + 1):
                    if vol_tag[vz, vy, vx] == 0:
                        vol_data[vz, vy, vx, :] = feat[ind, :]

    # gaussian filter
    if sigma > 0:
        for i in range(vol_data.shape[-1]):
            vol_data[:,:,:,i] = sp.ndimage.filters.gaussian_filter(vol_data[:,:,:,i], sigma=sigma, truncate=2)

    return vol_data, vol_tag


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



###############################################################################
# start the main script

g_prefix = ''

if g_external_test:
    input_test_hdf = h5py.File(os.path.join(g_main_dir, g_input_test_hd_file), 'r')
    output_test_hdf = h5py.File(os.path.join(g_main_dir, g_output_test_hd_file), 'w')
else:
    # open input hd5
    input_train_hdf = h5py.File(os.path.join(g_main_dir, g_input_train_hd_file), 'r')
    input_val_hdf = h5py.File(os.path.join(g_main_dir, g_input_val_hd_file), 'r')
    input_test_hdf = h5py.File(os.path.join(g_main_dir, g_input_test_hd_file), 'r')

    # create output hd5
    output_train_hdf = h5py.File(os.path.join(g_main_dir, g_output_train_hd_file), 'w')
    output_val_hdf = h5py.File(os.path.join(g_main_dir, g_output_val_hd_file), 'w')
    output_test_hdf = h5py.File(os.path.join(g_main_dir, g_output_test_hd_file), 'w')

# create output csv
output_csv_fp = open(os.path.join(g_main_dir, g_output_csv), 'w')
output_csv = csv.writer(output_csv_fp, delimiter=',')
output_csv.writerow(g_csv_header)


###############################################################################
# generate 3D for ligand and complex

feat_tool_str = g_feat_tool_list[g_feat_tool_ind]
feat_type_str = g_feat_type_list[g_feat_type_ind]
feat_pdbbind_str = g_feat_pdbbind_type_list2[g_feat_pdbbind_type_ind]

if g_external_test:
    input_hdfs = [input_test_hdf]
    output_hdfs = [output_test_hdf]
    output_prefixes = [g_output_test_hd_file[:-4]]
    traintest_splits = [2]
else:
    input_hdfs = [input_train_hdf, input_val_hdf, input_test_hdf]
    output_hdfs = [output_train_hdf, output_val_hdf, output_test_hdf]
    output_prefixes = [g_output_train_hd_file[:-4], g_output_val_hd_file[:-4], g_output_test_hd_file[:-4]]
    traintest_splits = [0, 1, 2]

for input_hdf, output_hdf, output_prefix, split in zip(input_hdfs, output_hdfs, output_prefixes, traintest_splits):
    for lig_id in input_hdf.keys():
        #if len(g_prefix) > 0 and not lig_id.startswith(g_prefix):
        #g_prefixcontinue

        feat_tool_list = input_hdf[lig_id]
        if not feat_tool_str in feat_tool_list:
            continue
        feat_type_list = feat_tool_list[feat_tool_str]
        if not feat_type_str in feat_type_list:
            continue
        
        feat_pdbbind_list = feat_type_list[feat_type_str]
        if not feat_pdbbind_str in feat_pdbbind_list:
            continue
            
        print("processing %s" % lig_id)
        
        if g_feat_pdbbind_type_ind == 1:
            feat_data_0 = feat_pdbbind_list[feat_pdbbind_str]
            for n in range(1,11):  # assuming pose1 to pose10
                if not str(n) in feat_data_0:
                    continue

                feat_data = feat_data_0[str(n)]
                input_data = feat_data[g_feat_data_str]
                input_radii = None
                if g_3D_atom_radii:
                    input_radii = feat_data.attrs['van_der_waals']
                input_affinity = input_hdf[lig_id].attrs['affinity']

                input_xyx = input_data[:,0:3]
                input_feat = input_data[:,3:]

                output_3d_data = get_3D_all2(input_xyx, input_feat, g_3D_dim, g_3D_relative_size, g_3D_size_angstrom, input_radii, g_3D_atom_radius, g_3D_sigma)
                print(input_data.shape, 'is converted into ', output_3d_data.shape)

                lig_id_pose = lig_id + '_' + str(n)
                if g_output_hd_compress:
                    output_hdf.create_dataset(lig_id_pose, data=output_3d_data, shape=output_3d_data.shape, dtype='float32', compression='lzf')
                else:
                    output_hdf.create_dataset(lig_id_pose, data=output_3d_data, shape=output_3d_data.shape, dtype='float32')
                output_hdf[lig_id_pose].attrs['affinity'] = input_affinity
            
                lig_prefix = '%d/%s/%s' % (split, output_prefix, lig_id_pose)
                output_csv.writerow([lig_id_pose, lig_prefix, input_affinity, split, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        else:
            feat_data = feat_pdbbind_list[feat_pdbbind_str]
            input_data = feat_data[g_feat_data_str]
            input_radii = None
            if g_3D_atom_radii:
                input_radii = feat_data.attrs['van_der_waals']
            input_affinity = input_hdf[lig_id].attrs['affinity']

            input_xyx = input_data[:,0:3]
            input_feat = input_data[:,3:]
            
            output_3d_data = get_3D_all2(input_xyx, input_feat, g_3D_dim, g_3D_relative_size, g_3D_size_angstrom, input_radii, g_3D_atom_radius, g_3D_sigma)
            print(input_data.shape, 'is converted into ', output_3d_data.shape)

            #dgroup = output_hdf.create_group(lig_id)
            if g_output_hd_compress:
                output_hdf.create_dataset(lig_id, data=output_3d_data, shape=output_3d_data.shape, dtype='float32', compression='lzf')
            else:
                output_hdf.create_dataset(lig_id, data=output_3d_data, shape=output_3d_data.shape, dtype='float32')
            output_hdf[lig_id].attrs['affinity'] = input_affinity
            
            lig_prefix = '%d/%s/%s' % (split, output_prefix, lig_id)
            output_csv.writerow([lig_id, lig_prefix, input_affinity, split, 0, 0, 0, 0, 0, 0, 0, 0, 0])

output_csv_fp.close()
if g_external_test:
    output_test_hdf.close()
else:
    output_train_hdf.close()
    output_val_hdf.close()
    output_test_hdf.close()

