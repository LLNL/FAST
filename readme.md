# Fusion models for Atomic and molecular STructures (FAST)

Predicting accurate protein-ligand binding affinity is important in drug discovery. This code implements fusion network model to benefit from Spatial Grach CNN and 3D CNN models to improve the binding affinity prediction. The code is written in python with Tensorflow and Pytorch.  

 

## Getting Started

### Prerequisites

- [Tensorflow 1.14 or higher (but not 2.X](https://www.tensorflow.org)
- [PyTorch 1.4 or higher](https://pytorch.org)
- [PyTorch Geometric Feature)](https://github.com/rusty1s/pytorch_geometric)
- [rdkit](rdkit.org)
- [pybel](https://github.com/pybel/pybel)
- [pdbfixer](https://github.com/openmm/pdbfixer)
- [tfbio](https://gitlab.com/cheminfIBB/tfbio)


### Running the application

#### Data format

The implemented networks use a 3D atomic representation as input data in a Hierarchical Data Format (HDF5). 
Each complex/pocket data is comprised of a list of atoms with their features including 3D coordinates of the atoms (x, y, z) and associated features such as atomic number and charges. For more detail, please refer to the paper in the Citing LIST section.  


#### 3D-CNN

To train or test 3D-CNN, run model/3dcnn/main_3dcnn_pdbbind.py. 
Here is an example comand to test a pre-trained 3D-CNN model:

python main_3dcnn_pdbbind.py --main-dir "pdbbind_3dcnn" --model-subdir "pdbbind2016_refined" --run-mode 5 --external-hdftype 3 --external-testhdf "eval_set.hdf" --external-featprefix "eval_3dcnn" --external-dir "pdbbind_2019"



#### SG-CNN

To train or test SG-CNN, run model/sgcnn/src/train.py or model/sgcnn/src/test.py. 

For an example training script, see model/sgcnn/scripts/train_pybel_pdbbind_2016_general_refined.sh 


#### Fusion

To train or test fusion model, run model/fusion/main_fusion_pdbbind.py

python main_fusion_pdbbind.py --main-dir "pdbbind_fusion" --fusionmodel-subdir "pdbbind2016_fusion" --run-mode 3 --external-csvfile "eval_3dcnn.csv" --external-3dcnn-featfile "eval_3dcnn_fc10.npy" --external-sgcnn-featfile "eval_sgcnn_feat.npy" --external-outprefix "eval_fusion" --external-dir "pdbbind_2019"


#### Pre-trained weights (checkpoint files)

We trained all of the networks above on [pdbbind 2016 datasets](http://www.pdbbind.org.cn). Particularly, we used general and refined datasets for training and validation, and evaluated the model on the core set (see sample_data/core_test.hdf). 

The checkpoint files for the models are made available under [the Creative Commons BY 4.0 license](https://creativecommons.org/licenses/by/4.0/). To download the files, refer to [this public ftp site](ftp://gdo-bioinformatics.ucllnl.org/fast/pdbbind2016_model_checkpoints/). LLNL-MI-813373



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

Derek Jones, Hyojin Kim, Xiaohua Zhang, Adam Zemla, William D. Bennett, Dan Kirshner, Sergio Wong, Felice
Lightstone, and Jonathan E. Allen, "Improved Protein-ligand Binding Affinity Prediction with Structure-Based Deep Fusion Inference", arxiv 2020. 



## License
FAST is distributed under the terms of the MIT license. All new contributions must be made under this license.

SPDX-License-Identifier: MIT

LLNL-CODE-808183

