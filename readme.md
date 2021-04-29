# Fusion models for Atomic and molecular STructures (FAST)

Predicting accurate protein-ligand binding affinity is important in drug discovery. This code implements fusion network model to benefit from Spatial Grach CNN and 3D CNN models to improve the binding affinity prediction. The code is written in python with Tensorflow and Pytorch.  

 

## Getting Started

### Prerequisites

~~- Tensorflow 1.14 or higher~~
- [PyTorch 1.4 or higher](https://pytorch.org)
- [PyTorch Geometric Feature)](https://github.com/rusty1s/pytorch_geometric)
- [rdkit](rdkit.org) (optional)
- [pybel](https://github.com/pybel/pybel)  (optional)
- [pdbfixer](https://github.com/openmm/pdbfixer)  (optional)
- [tfbio](https://gitlab.com/cheminfIBB/tfbio)  (optional)


### Running the application

#### Data format

The implemented networks use a 3D atomic representation as input data in a Hierarchical Data Format (HDF5). 
Each complex/pocket data is comprised of a list of atoms with their features including 3D coordinates of the atoms (x, y, z) and associated features such as atomic number and charges. For more detail, please refer to the paper in the Citing LIST section.  


#### 3D-CNN

Note that the original 3D-CNN implementation used in the paper below has been moved to 3dcnn_tf. A new version using pytorch has been released in `model/3dcnn`


##### 3D-CNN tensorflow version (used in the paper)

To train or test 3D-CNN, run `model/3dcnn_tf/main_3dcnn_pdbbind.py`. 
Here is an example comand to test a pre-trained 3D-CNN model:

```
python main_3dcnn_pdbbind.py --main-dir "pdbbind_3dcnn" --model-subdir "pdbbind2016_refined" --run-mode 5 --external-hdftype 3 --external-testhdf "eval_set.hdf" --external-featprefix "eval_3dcnn" --external-dir "pdbbind_2019"
```

##### 3D-CNN pytorch version (new version)

In this new version, the voxelization process is done on GPU, which improves performance/speed-up. The new version is located in `model/3dcnn`

To train, run `model/3dcnn/main_train.py`
To test/evaluate, run `model/3dcnn/model_eval.py`

`model/data_reader.py` is a default data reader that reads our ML-HDF format described above. Please use your own data_reader to read your own format.


#### SG-CNN

To train or test SG-CNN, run `model/sgcnn/src/train.py` or `model/sgcnn/src/test.py`. 

For an example training script, see `model/sgcnn/scripts/train_pybel_pdbbind_2016_general_refined.sh`


#### Fusion

To train or test fusion model, run `model/fusion/main_fusion_pdbbind.py`

```
python main_fusion_pdbbind.py --main-dir "pdbbind_fusion" --fusionmodel-subdir "pdbbind2016_fusion" --run-mode 3 --external-csvfile "eval_3dcnn.csv" --external-3dcnn-featfile "eval_3dcnn_fc10.npy" --external-sgcnn-featfile "eval_sgcnn_feat.npy" --external-outprefix "eval_fusion" --external-dir "pdbbind_2019"
```

#### Pre-trained weights (checkpoint files)

We trained all of the networks above on [pdbbind 2016 datasets](http://www.pdbbind.org.cn). Particularly, we used general and refined datasets for training and validation, and evaluated the model on the core set (see sample_data/core_test.hdf). 

The checkpoint files for the models are made available under the Creative Commons BY 4.0 license. See the license section below for the terms of the license. The files can be found here: `ftp://gdo-bioinformatics.ucllnl.org/fast/pdbbind2016_model_checkpoints/`. 

#### PDBSpheres evaluation set

We make available the hold-out test set from the manuscript here: `sample_data/PDBSPHERES_EVAL_SET.csv`


## Contributing

To contribute to FAST, please send us a pull request. When you send your request, make develop 
the destination branch on the repository.
 


## Versioning
0.1



## Authors

FAST was created by Hyojin Kim (hkim@llnl.gov), Derek Jones (jones289@llnl.gov), Jonathan Allen (allen99@llnl.gov). 

### Other contributors
This project was supported by the American Heart Association (AHA) project (PI: Felice Lightstone). 



## Citing LIST

If you need to reference FAST in a publication, please cite the following paper:




Jones, D., Kim, H., Zhang, X., Zemla, A., Stevenson, G., Bennett, W. F. D., Kirshner, D., Wong, S. E., Lightstone, F. C., & Allen, J. E. (2021). Improved Protein-Ligand Binding Affinity Prediction with Structure-Based Deep Fusion Inference. Journal of Chemical Information and Modeling. https://doi.org/10.1021/acs.jcim.0c01306


```

@ARTICLE{Jones_Kim_improved_2021,
  title    = "Improved {Protein-Ligand} Binding Affinity Prediction with
              {Structure-Based} Deep Fusion Inference",
  author   = "Jones, Derek and Kim, Hyojin and Zhang, Xiaohua and Zemla, Adam
              and Stevenson, Garrett and Bennett, W F Drew and Kirshner, Daniel
              and Wong, Sergio E and Lightstone, Felice C and Allen, Jonathan E",
  journal  = "J. Chem. Inf. Model.",
  volume   =  61,
  number   =  4,
  pages    = "1583--1592",
  month    =  apr,
  year     =  2021,
  language = "en"
}

```



## License
FAST is distributed under the terms of the MIT license. All new contributions must be made under this license. See LICENSE in this directory for the terms of the license.
SPDX-License-Identifier: MIT
LLNL-CODE-808183

Checkpoint files are provided under [the Creative Commons BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). See LICENSE-CC-BY in this directory for the terms of the license.  
LLNL-MI-813373

