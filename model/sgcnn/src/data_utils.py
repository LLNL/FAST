################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Spatial Graph Convolutional Network data loading utilities
################################################################################


import torch
import os.path as osp

from glob import glob
import multiprocessing as mp
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import DataLoader as GeometricDataLoader
from tqdm import tqdm
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data, Batch

import h5py

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances


class PDBBindDataset(Dataset):
    def __init__(
        self,
        data_file,
        dataset_name,
        feature_type,
        preprocessing_type,
        use_docking=False,
        output_info=False,
        cache_data=True

    ):
        super(PDBBindDataset, self).__init__()
        self.dataset_name = dataset_name

        self.data_file = data_file
        self.feature_type = feature_type
        self.preprocessing_type = preprocessing_type
        self.use_docking = use_docking
        self.output_info = output_info
        self.cache_data = cache_data
        self.data_dict = {}  # will use this to store data once it has been computed if cache_data is True

        self.data_list = []  # will use this to store ids for data

        if self.use_docking:

            with h5py.File(data_file, "r") as f:

                for name in list(f):
                    # if the feature type (pybel or rdkit) not available, skip over it
                    if self.feature_type in list(f[name]):
                        affinity = np.asarray(f[name].attrs["affinity"]).reshape(1, -1)
                        if self.preprocessing_type in f[name][self.feature_type]:
                            if self.dataset_name in list(
                                f[name][self.feature_type][self.preprocessing_type]
                            ):
                                for pose in f[name][self.feature_type][
                                    self.preprocessing_type
                                ][self.dataset_name]:
                                    self.data_list.append((name, pose, affinity))

        else:

            with h5py.File(data_file, "r") as f:

                for name in list(f):
                    # if the feature type (pybel or rdkit) not available, skip over it
                    if self.feature_type in list(f[name]):
                        affinity = np.asarray(f[name].attrs["affinity"]).reshape(1, -1)

                        self.data_list.append(
                            (name, 0, affinity)
                        )  # Putting 0 for pose to denote experimental structure and to be consistent with docking data format

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):

        if self.cache_data:

            if item in self.data_dict.keys():
                return self.data_dict[item]

            else:
                pass       
 
        
        pdbid, pose, affinity = self.data_list[item]

        node_feats, coords = None, None
        with h5py.File(self.data_file, "r") as f:

            if (
                not self.dataset_name
                in f[
                    "{}/{}/{}".format(
                        pdbid, self.feature_type, self.preprocessing_type
                    )
                ].keys()
            ):
                print(pdbid)
                return None

            if self.use_docking:
                # TODO: the next line will cuase runtime error because not selelcting poses
                data = f[
                    "{}/{}/{}/{}".format(
                        pdbid,
                        self.feature_type,
                        self.preprocessing_type,
                        self.dataset_name,
                    )
                ][pose]["data"]
                vdw_radii = (
                    f[

                        "{}/{}/{}/{}".format(
                            pdbid,
                            self.feature_type,
                            self.preprocessing_type,
                            self.dataset_name,
                        )
                    ][pose]
                    .attrs["van_der_waals"]
                    .reshape(-1, 1)
                )

            else:
                data = f[
                    "{}/{}/{}/{}".format(
                        pdbid,
                        self.feature_type,
                        self.preprocessing_type,
                        self.dataset_name,
                    )
                ]["data"]
                vdw_radii = (
                    f[

                        "{}/{}/{}/{}".format(
                            pdbid,
                            self.feature_type,
                            self.preprocessing_type,
                            self.dataset_name,
                        )

                    ]
                    .attrs["van_der_waals"]
                    .reshape(-1, 1)
                )

            if self.feature_type == "pybel":
                coords = data[:, 0:3]
                node_feats = np.concatenate([vdw_radii, data[:, 3:22]], axis=1)

            else:
                raise NotImplementedError

        # account for the vdw radii in distance cacluations (consider each atom as a sphere, distance between spheres)

        dists = pairwise_distances(coords, metric="euclidean")

        edge_index, edge_attr = dense_to_sparse(torch.from_numpy(dists).float())

        x = torch.from_numpy(node_feats).float()

        y = torch.FloatTensor(affinity).view(-1, 1)
        data = Data(
            x=x, edge_index=edge_index, edge_attr=edge_attr.view(-1, 1), y=y
        )


        if self.cache_data:

            if self.output_info:
                self.data_dict[item] = (pdbid, pose, data)

            else:
                self.data_dict[item] = data

            return self.data_dict[item]

        else:
            if self.output_info:
                return (pdbid, pose, data)
            else:
                return data


def worker_init_fn(worker_id):
    np.random.seed(int(0))


def test():

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--docking-2007", default=False, action="store_true")
    parser.add_argument("--exp-2007", default=False, action="store_true")
    parser.add_argument("--exp-2016", default=False, action="store_true")
    args = parser.parse_args()

    from torch.utils.data import ConcatDataset

    docking_core_2007_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2007_with_docking/core.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="docking",
        use_docking=True,
    )
    docking_refined_2007_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2007_with_docking/refined.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="docking",
        use_docking=True,
    )

    exp_core_2007_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2007_with_docking/core.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="pdbbind",
        use_docking=False,
    )
    exp_refined_2007_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2007_with_docking/refined.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="pdbbind",
        use_docking=False,
    )

    exp_core_2016_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2016_pybel_processed/core.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="pdbbind",
        use_docking=False,
    )
    exp_refined_2016_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2016_pybel_processed/refined.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="pdbbind",
        use_docking=False,
    )
    exp_general_2016_dataset = PDBBindDataset(
        data_file="/p/lscratchh/jones289/data/pdbbind_2016_pybel_processed/general.hdf",
        feature_type="pybel",
        preprocessing_type="processed",
        dataset_name="pdbbind",
        use_docking=False,
    )

    if args.exp_2016:
        dataset_2016 = ConcatDataset(
            [exp_core_2016_dataset, exp_refined_2016_dataset, exp_general_2016_dataset]
        )
        dataloader_2016 = GeometricDataLoader(
            dataset_2016,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )
        print("{} experimental complexes in 2016 dataset".format(len(dataset_2016)))

        for batch in tqdm(dataloader_2016, desc="2016 experimental data"):
            pass

    if args.exp_2007:
        dataset_2007 = ConcatDataset([exp_core_2007_dataset, exp_refined_2007_dataset])
        dataloader_2007 = GeometricDataLoader(
            dataset_2007,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )
        print("{} experimental complexes in 2007 dataset".format(len(dataset_2007)))

        for batch in tqdm(dataloader_2007, desc="2007 experimental data"):
            pass

    if args.docking_2007:
        docking_2007_dataset = ConcatDataset(
            [docking_core_2007_dataset, docking_refined_2007_dataset]
        )
        docking_2007_dataloader = GeometricDataLoader(
            docking_2007_dataset,
            batch_size=4,
            shuffle=True,
            num_workers=4,
            worker_init_fn=worker_init_fn,
        )
        print("{} docking complexes in 2007 dataset".format(len(docking_2007_dataset)))

        for batch in tqdm(docking_2007_dataloader, desc="2007 docking data"):
            pass


if __name__ == "__main__":

    test()
