################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Spatial Graph Convolutional Network training script
################################################################################


import os
import itertools
from glob import glob
import multiprocessing as mp
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import init
from torch.optim import Adam, lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from torch_geometric.data import Data, Batch
from torch_geometric.data import DataLoader as GeometricDataLoader
from torch_geometric.nn import DataParallel as GeometricDataParallel
from torch_geometric.data import DataListLoader
from data_utils import PDBBindDataset
from model import PotentialNetParallel, GraphThreshold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, SubsetRandomSampler


import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--checkpoint", type=bool, default=False, help="boolean flag for checkpoints"
)
parser.add_argument(
    "--checkpoint-dir", default=os.getcwd(), help="path to store model checkpoints"
)
parser.add_argument(
    "--checkpoint-iter", default=10, type=int, help="number of epochs per checkpoint"
)
parser.add_argument("--epochs", default=100, type=int, help="number of training epochs")
parser.add_argument(
    "--num-workers", default=24, type=int, help="number of workers for dataloader"
)
parser.add_argument(
    "--batch-size", default=32, type=int, help="batch size to use for training"
)
parser.add_argument(
    "--lr", default=1e-3, type=float, help="learning rate to use for training"
)
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
parser.add_argument(
    "--dataset-name", type=str, required=True
)  # NOTE: this should probably just consist of a set of choices




parser.add_argument("--covalent-gather-width", type=int, default=128)
parser.add_argument("--non-covalent-gather-width", type=int, default=128)
parser.add_argument("--covalent-k", type=int, default=1)
parser.add_argument("--non-covalent-k", type=int, default=1)
parser.add_argument("--covalent-threshold", type=float, default=1.5)
parser.add_argument("--non-covalent-threshold", type=float, default=7.5)
parser.add_argument("--train-data", type=str, required=True, nargs="+")
parser.add_argument("--val-data", type=str, required=True, nargs="+")
parser.add_argument("--use-docking", default=False, action="store_true")
args = parser.parse_args()

# seed all random number generators and set cudnn settings for deterministic: https://github.com/rusty1s/pytorch_geometric/issues/217
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # NOTE: https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
os.environ["PYTHONHASHSEED"] = "0"


def worker_init_fn(worker_id):
    np.random.seed(int(0))


def collate_fn_none_filter(batch):
    return [x for x in batch if x is not None]


def train():

    # set the input channel dims based on featurization type
    if args.feature_type == "pybel":
        feature_size = 20
    else:
        feature_size = 75

    print("found {} datasets in input train-data".format(len(args.train_data)))
    train_dataset_list = []
    val_dataset_list = []

    for data in args.train_data:
        train_dataset_list.append(
            PDBBindDataset(
                data_file=data,
                dataset_name=args.dataset_name,
                feature_type=args.feature_type,
                preprocessing_type=args.preprocessing_type,
                output_info=True,
                use_docking=args.use_docking,
            )
        )

    for data in args.val_data:
        val_dataset_list.append(
            PDBBindDataset(
                data_file=data,
                dataset_name=args.dataset_name,
                feature_type=args.feature_type,
                preprocessing_type=args.preprocessing_type,
                output_info=True,
                use_docking=args.use_docking,
            )
        )

    train_dataset = ConcatDataset(train_dataset_list)
    val_dataset = ConcatDataset(val_dataset_list)

    train_dataloader = DataListLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )  # just to keep batch sizes even, since shuffling is used

    val_dataloader = DataListLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    tqdm.write("{} complexes in training dataset".format(len(train_dataset)))
    tqdm.write("{} complexes in validation dataset".format(len(val_dataset)))

    model = GeometricDataParallel(
        PotentialNetParallel(
            in_channels=feature_size,
            out_channels=1,
            covalent_gather_width=args.covalent_gather_width,
            non_covalent_gather_width=args.non_covalent_gather_width,
            covalent_k=args.covalent_k,
            non_covalent_k=args.non_covalent_k,
            covalent_neighbor_threshold=args.covalent_threshold,
            non_covalent_neighbor_threshold=args.non_covalent_threshold,
        )
    ).float()

    model.train()
    model.to(0)
    tqdm.write(str(model))
    tqdm.write(
        "{} trainable parameters.".format(
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    tqdm.write(
        "{} total parameters.".format(sum(p.numel() for p in model.parameters()))
    )

    criterion = nn.MSELoss().float()
    optimizer = Adam(model.parameters(), lr=args.lr)

    best_checkpoint_dict = None
    best_checkpoint_epoch = 0
    best_checkpoint_step = 0
    best_checkpoint_r2 = -9e9
    step = 0
    for epoch in range(args.epochs):
        losses = []
        for batch in tqdm(train_dataloader):
            batch = [x for x in batch if x is not None]
            if len(batch) < 1:
                print("empty batch, skipping to next batch")
                continue
            optimizer.zero_grad()

            data = [x[2] for x in batch]
            y_ = model(data)
            y = torch.cat([x[2].y for x in batch])

            loss = criterion(y.float(), y_.cpu().float())
            losses.append(loss.cpu().data.item())
            loss.backward()

            y_true = y.cpu().data.numpy()
            y_pred = y_.cpu().data.numpy()

            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)

            pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
            spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))

            tqdm.write(
                "epoch: {}\tloss:{:0.4f}\tr2: {:0.4f}\t pearsonr: {:0.4f}\tspearmanr: {:0.4f}\tmae: {:0.4f}\tpred stdev: {:0.4f}"
                "\t pred mean: {:0.4f} \tcovalent_threshold: {:0.4f} \tnon covalent threshold: {:0.4f}".format(
                    epoch,
                    loss.cpu().data.numpy(),
                    r2,
                    float(pearsonr[0]),
                    float(spearmanr[0]),
                    float(mae),
                    np.std(y_pred),
                    np.mean(y_pred),
                    model.module.covalent_neighbor_threshold.t.cpu().data.item(),
                    model.module.non_covalent_neighbor_threshold.t.cpu().data.item(),
                )
            )

            if args.checkpoint:
                if step % args.checkpoint_iter == 0:
                    checkpoint_dict = checkpoint_model(
                        model,
                        val_dataloader,
                        epoch,
                        step,
                        args.checkpoint_dir
                        + "/model-epoch-{}-step-{}.pth".format(epoch, step),
                    )
                    if checkpoint_dict["validate_dict"]["r2"] > best_checkpoint_r2:
                        best_checkpoint_step = step
                        best_checkpoint_epoch = epoch
                        best_checkpoint_r2 = checkpoint_dict["validate_dict"]["r2"]
                        best_checkpoint_dict = checkpoint_dict

            optimizer.step()
            step += 1

        if args.checkpoint:
            checkpoint_dict = checkpoint_model(
                model,
                val_dataloader,
                epoch,
                step,
                args.checkpoint_dir + "/model-epoch-{}-step-{}.pth".format(epoch, step),
            )
            if checkpoint_dict["validate_dict"]["r2"] > best_checkpoint_r2:
                best_checkpoint_step = step
                best_checkpoint_epoch = epoch
                best_checkpoint_r2 = checkpoint_dict["validate_dict"]["r2"]
                best_checkpoint_dict = checkpoint_dict

    if args.checkpoint:
        # once broken out of the loop, save last model
        checkpoint_dict = checkpoint_model(
            model,
            val_dataloader,
            epoch,
            step,
            args.checkpoint_dir + "/model-epoch-{}-step-{}.pth".format(epoch, step),
        )

        if checkpoint_dict["validate_dict"]["r2"] > best_checkpoint_r2:
            best_checkpoint_step = step
            best_checkpoint_epoch = epoch
            best_checkpoint_r2 = checkpoint_dict["validate_dict"]["r2"]
            best_checkpoint_dict = checkpoint_dict

    if args.checkpoint:
        torch.save(best_checkpoint_dict, args.checkpoint_dir + "/best_checkpoint.pth")
    print(
        "best training checkpoint epoch {}/step {} with r2: {}".format(
            best_checkpoint_epoch, best_checkpoint_step, best_checkpoint_r2
        )
    )


def validate(model, val_dataloader):

    model.eval()

    y_true = []
    y_pred = []
    pdbid_list = []
    pose_list = []

    for batch in tqdm(val_dataloader):
        data = [x[2] for x in batch if x is not None]
        y_ = model(data)
        y = torch.cat([x[2].y for x in batch])

        pdbid_list.extend([x[0] for x in batch])
        pose_list.extend([x[1] for x in batch])
        y_true.append(y.cpu().data.numpy())
        y_pred.append(y_.cpu().data.numpy())

    y_true = np.concatenate(y_true).reshape(-1, 1)
    y_pred = np.concatenate(y_pred).reshape(-1, 1)

    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    mae = mean_absolute_error(y_true=y_true, y_pred=y_pred)
    mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
    pearsonr = stats.pearsonr(y_true.reshape(-1), y_pred.reshape(-1))
    spearmanr = stats.spearmanr(y_true.reshape(-1), y_pred.reshape(-1))

    tqdm.write(
        str(
            "r2: {}\tmae: {}\tmse: {}\tpearsonr: {}\t spearmanr: {}".format(
                r2, mae, mse, pearsonr, spearmanr
            )
        )
    )
    model.train()
    return {
        "r2": r2,
        "mse": mse,
        "mae": mae,
        "pearsonr": pearsonr,
        "spearmanr": spearmanr,
        "y_true": y_true,
        "y_pred": y_pred,
        "pdbid": pdbid_list,
        "pose": pose_list,
    }


def checkpoint_model(model, dataloader, epoch, step, output_path):
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    validate_dict = validate(model, dataloader)
    model.train()

    checkpoint_dict = {
        "model_state_dict": model.state_dict(),
        "args": vars(args),
        "step": step,
        "epoch": epoch,
        "validate_dict": validate_dict,
    }

    torch.save(checkpoint_dict, output_path)

    # return the computed metrics so it can be used to update the training loop
    return checkpoint_dict


def main():
    train()


if __name__ == "__main__":
    main()
