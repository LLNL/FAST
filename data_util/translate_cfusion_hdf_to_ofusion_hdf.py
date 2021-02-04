################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# This python code contains utilities to translate hdf5 files for 'Coherent-Fusion' and 'Original-Fusion' formats.
################################################################################


import h5py
import pandas as pd
from tqdm import tqdm

def translate_cfusion_to_ofusion(args):
    
    # check to see if we have a complex-subset-file that was specified
    complex_list = []
    if args.complex_subset_file:
        complex_list = pd.read_csv(args.complex_subset_file, header=None)[0].values


    with h5py.File(args.ofusion_file, 'w') as ofusion_handle:

        for cfusion_file in args.cfusion_files:

            with h5py.File(cfusion_file, 'r') as cfusion_handle:
                for complex_name, complex_grp in tqdm(cfusion_handle[args.cfusion_task_type].items()):
                    if complex_name in complex_list:
                        ofusion_complex_grp = ofusion_handle.require_group(f'{complex_name}/pybel/processed/pdbbind')
                        ofusion_handle[f'{complex_name}'].attrs['affinity'] = complex_grp.attrs['affinity']
                        ofusion_complex_grp.create_dataset('data', data=complex_grp['pybel/processed/pdbbind_sgcnn/data0'])
                        ofusion_complex_grp.attrs['van_der_waals'] = complex_grp['pybel/processed/pdbbind_sgcnn'].attrs['van_der_waals']
                    else:
                        #nothing to do so just pass and continue over to the next complex 
                        pass 


def main(args):
    translate_cfusion_to_ofusion(args)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfusion-files', help='file path name for the coherent-fusion model, treated as a list in the python code so can also specify multiple files, useful if want to merge multiple files into one', nargs='+', required=True)
    parser.add_argument('--cfusion-task-type', choices=['regression', 'classification'], help='what type of task to access in the cfusion file, for reading from or writing to', default='regression')
    parser.add_argument('--ofusion-file', help='file path name for the original-fusion model')
    parser.add_argument('--complex-subset-file', help='this is a csv file that contains a list of complexes to include in the output hdf file')
    args = parser.parse_args()
    main(args)

