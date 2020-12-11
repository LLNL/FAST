################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Spatial Graph Convolutional Network evaluation script
################################################################################


import os
import os.path as osp
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import h5py
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from tqdm import tqdm
from glob import glob
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.data import Data, Batch, DataListLoader
from data_utils import PDBBindDataset
from model import PotentialNetParallel


def test(args):

    if torch.cuda.is_available():

        model_train_dict = torch.load(args.checkpoint)

    else:
        model_train_dict = torch.load(args.checkpoint, map_location=torch.device('cpu'))

    '''
    model = GeometricDataParallel(
        PotentialNetParallel(
            in_channels=20,
            out_channels=1,
            covalent_gather_width=model_train_dict["args"]["covalent_gather_width"],
            non_covalent_gather_width=model_train_dict["args"][
                "non_covalent_gather_width"
            ],
            covalent_k=model_train_dict["args"]["covalent_k"],
            non_covalent_k=model_train_dict["args"]["non_covalent_k"],
            covalent_neighbor_threshold=model_train_dict["args"]["covalent_threshold"],
            non_covalent_neighbor_threshold=model_train_dict["args"][
                "non_covalent_threshold"
            ],
        )
    ).float()
    
    '''
    model = PotentialNetParallel(
            in_channels=20,
            out_channels=1,
            covalent_gather_width=model_train_dict["args"]["covalent_gather_width"],
            non_covalent_gather_width=model_train_dict["args"][
                "non_covalent_gather_width"
            ],
            covalent_k=model_train_dict["args"]["covalent_k"],
            non_covalent_k=model_train_dict["args"]["non_covalent_k"],
            covalent_neighbor_threshold=model_train_dict["args"]["covalent_threshold"],
            non_covalent_neighbor_threshold=model_train_dict["args"][
                "non_covalent_threshold"
            ],
        ).float()
    
    model_module = torch.nn.Module()
    model_module.add_module('module', model)
    model = model_module

    print(model_module, model)
    

    model.load_state_dict(model_train_dict["model_state_dict"])

    dataset_list = []
    
    # because the script allows for multiple datasets, we iterate over the list of files to build one combined dataset object
    for data in args.test_data:
        dataset_list.append(
            PDBBindDataset(
                data_file=data,
                dataset_name=args.dataset_name,
                feature_type=args.feature_type,
                preprocessing_type=args.preprocessing_type,
                output_info=True,
                cache_data=False,
                use_docking=args.use_docking,
            )
        )

    dataset = ConcatDataset(dataset_list)
    print("{} complexes in dataset".format(len(dataset)))

    dataloader = DataListLoader(dataset, batch_size=args.batch_size, shuffle=False)

    model.eval()
    if torch.cuda.is_available():
    
        model.cuda()

    if args.print_model:
        print(model)
    print("{} total parameters.".format(sum(p.numel() for p in model.parameters())))

    if not os.path.exists(args.output):
        os.makedirs(args.output)


    output_f = "{}/{}".format(args.output, args.output_file_name) 

    with h5py.File(output_f, "w") as f:

        for batch in tqdm(dataloader):

            batch = [x for x in batch if x is not None]
            if len(batch) < 1:
                continue

            for item in batch:
                name = item[0]
                pose = item[1]
                data = item[2]

                name_grp = f.require_group(str(name))

                name_pose_grp = name_grp.require_group(str(pose))

                y = data.y

                name_pose_grp.attrs["y_true"] = y

                (
                    covalent_feature,
                    non_covalent_feature,
                    pool_feature,
                    fc0_feature,
                    fc1_feature,
                    y_,
                ) = model.module(
                    Batch().from_data_list([data]), return_hidden_feature=True
                )

                name_pose_grp.attrs["y_pred"] = y_.cpu().data.numpy()
                hidden_features = np.concatenate(
                    (
                        covalent_feature.cpu().data.numpy(),
                        non_covalent_feature.cpu().data.numpy(),
                        pool_feature.cpu().data.numpy(),
                        fc0_feature.cpu().data.numpy(),
                        fc1_feature.cpu().data.numpy(),
                    ),
                    axis=1,
                )

                name_pose_grp.create_dataset(
                    "hidden_features",
                    (hidden_features.shape[0], hidden_features.shape[1]),
                    data=hidden_features,
                )

    
def main(args):
    test(args)


if __name__ == "__main__": 

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="path to model checkpoint")

    parser.add_argument(
        "--preprocessing-type",
        type=str,
        choices=["raw", "processed"],
        help="idicate raw pdb or (chimera) processed",
        required=True,
    )
    parser.add_argument(
        "--feature-type",
        type=str,
        choices=["pybel", "rdkit"],
        help="indicate pybel (openbabel) or rdkit features",
        required=True,
    )
    parser.add_argument("--dataset-name", type=str, required=True)

    parser.add_argument(
        "--batch-size", type=int, default=1, help="batch size to use for dataloader"
    )

    parser.add_argument(
        "--num-workers", default=24, type=int, help="number of workers for dataloader"
    )

    parser.add_argument("--test-data", nargs="+", required=True)
    parser.add_argument("--output", help="path to output directory")
    parser.add_argument(
        "--use-docking",
        help="flag to indicate if dataset contains docking info",
        default=False,
        action="store_true",
    )
    parser.add_argument("--output-file-name", help="output file name", required=True)
    parser.add_argument("--print-model", action="store_true", help="bool flag to determine whether to print the model")
    args = parser.parse_args()


    main(args)
