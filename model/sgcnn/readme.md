# Spatial Graph Neural Network (SG-CNN)

This directory contains the source code for the SG-CNN and is organized as:

        * src: train.py, test.py, ggcnn.py, model.py, data_utils.py
        * scripts: contains various scripts for training different models

In order to train a model and later evaluate it, a concrete example is provided below:

```
    python train.py --checkpoint-dir=${checkpoint_dir}--num-workers=8 --batch-size=8 --preprocessing-type=processed --feature-type=pybel --epochs=300 --lr=1e-3 --covalent-threshold=1.5 --non-covalent-threshold=4.5 --covalent-gather-width=16 --covalent-k=2 --non-covalent-gather-width=12 --non-covalent-k=2 --checkpoint=True --checkpoint-iter=100 --train-data /g/g13/jones289/data/pdbbind_2016_pybel_processed_train_val_test/general_train.hdf /g/g13/jones289/data/pdbbind_2016_pybel_processed_train_val_test/refined_train.hdf --val-data /g/g13/jones289/data/pdbbind_2016_pybel_processed_train_val_test/general_val.hdf /g/g13/jones289/data/pdbbind_2016_pybel_processed_train_val_test/refined_val.hdf --dataset-name pdbbind


python test.py --checkpoint "${checkpoint_dir}/best_checkpoint.pth"  --preprocessing-type=processed --feature-type=pybel --dataset-name pdbbind --num-workers 1 --output "${output_dir}" --test-data $test_data_path;

```
