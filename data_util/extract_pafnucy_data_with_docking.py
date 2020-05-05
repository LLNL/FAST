################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Generate ML-HDF for sgcnn and 3dcnn
################################################################################


import os
from tfbio.data import Featurizer
import numpy as np
import h5py
import argparse
import pybel
import warnings
#from data_generator.atomfeat_util import read_pdb, rdkit_atom_features, rdkit_atom_coords
#from data_generator.chem_info import g_atom_vdw_ligand, g_atom_vdw_protein
import xml.etree.ElementTree as ET
from rdkit.Chem.rdmolfiles import MolFromMol2File
import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem
from pybel import Atom
import pandas as pd
from tqdm import tqdm
from glob import glob

ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)

# TODO: compute rdkit features and store them in the output hdf5 file
# TODO: instead of making a file for each split, squash into one?


# TODO: not sure setting these to defaults is a good idea...
parser = argparse.ArgumentParser()
parser.add_argument("--input-pdbbind", default="/g/g13/jones289/data/raw_data/v2007")
parser.add_argument("--input-docking", default="/g/g13/jones289/data/raw_data/pdbbind_2007_docking_output")
parser.add_argument("--use-docking", default=False, action="store_true")
parser.add_argument("--use-exp", default=False, action="store_true")
parser.add_argument("--output", default="/g/g13/jones289/data/pdbbind_2007_with_docking")
parser.add_argument("--metadata", default="/g/g13/jones289/data/pdbbind/2007_affinity_data_cleaned.csv")
args = parser.parse_args()


def parse_element_description(desc_file):
    element_info_dict = {}
    element_info_xml = ET.parse(desc_file)
    for element in element_info_xml.getiterator():
        if "comment" in element.attrib.keys():
            continue
        else:
            element_info_dict[int(element.attrib["number"])] = element.attrib

    return element_info_dict


def parse_mol_vdw(mol, element_dict):
    vdw_list = []

    if isinstance(mol, pybel.Molecule):
        for atom in mol.atoms:
            # NOTE: to be consistent between featurization methods, throw out the hydrogens
            if int(atom.atomicnum) == 1:
                continue
            if int(atom.atomicnum) == 0:
                continue
            else:
                vdw_list.append(float(element_dict[atom.atomicnum]["vdWRadius"]))

    elif isinstance(mol, rdkit.Chem.rdchem.Mol):
        for atom in mol.GetAtoms():
            # NOTE: to be consistent between featurization methods, throw out the hydrogens
            if int(atom.GetAtomicNum()) == 1:
                continue
            else:
                vdw_list.append(float(element_dict[atom.GetAtomicNum()]["vdWRadius"]))
    else:
        raise RuntimeError("must provide a pybel mol or an RDKIT mol")

    return np.asarray(vdw_list)


def featurize_pybel_complex(ligand_mol, pocket_mol, name, dataset_name):

    featurizer = Featurizer()
    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge') 

    # get ligand features
    ligand_coords, ligand_features = featurizer.get_features(ligand_mol, molcode=1)

    if not (ligand_features[:, charge_idx] != 0).any():  # ensures that partial charge on all atoms is non-zero?
        raise RuntimeError("invalid charges for the ligand {} ({} set)".format(name, dataset_name))  

    # get processed pocket features
    pocket_coords, pocket_features = featurizer.get_features(pocket_mol, molcode=-1)
    if not (pocket_features[:, charge_idx] != 0).any():
        raise RuntimeError("invalid charges for the pocket {} ({} set)".format(name, dataset_name))   

    # center the coordinates on the ligand coordinates
    centroid_ligand = ligand_coords.mean(axis=0)
    ligand_coords -= centroid_ligand

    pocket_coords -= centroid_ligand
    data = np.concatenate((np.concatenate((ligand_coords, pocket_coords)), 
                                np.concatenate((ligand_features, pocket_features))), axis=1) 

    return data



def main(): 
 
    affinity_data = pd.read_csv(args.metadata)

    element_dict = parse_element_description("data_util/elements.xml")
 
    failure_dict = {"name": [], "partition": [], "set": [], "error": []}

    for dataset_name, data in tqdm(affinity_data.groupby('set')):
        print("found {} complexes in {} set".format(len(data), dataset_name))

        if not os.path.exists(args.output):
            os.makedirs(args.output) 

        with h5py.File('%s/%s.hdf' % (args.output, dataset_name), 'w') as f:

            for idx, row in tqdm(data.iterrows(), total=data.shape[0]):

                name = row['name']

                affinity = row['-logKd/Ki']

                receptor_path = row['receptor_path']
            

                '''
                    here is where the ligand(s) for both the experimental structure and the docking data need to be loaded.
                    * In order to do this, need an input path for both the experimental data as well as the docking data
                    * For docking data:
                        > Need to know how many poses there are, potentially up to 10 but not always the case
                        > May not have ligand/pocket data for names, need to handle this possibility

                    ######################################################################################################



                            BREAK THE MAIN LOOP INTO TWO PARTS....PROCESS DOCKING and PROCESS CRYSTAL STRUCTURES



                    ######################################################################################################

                '''

                ############################## CREATE THE PDB GROUP ##################################################
                # this is here in order to ensure any dataset that is created has passed the quality check, i.e. no failed complexes enter the output file

                
                grp = f.create_group(str(name))
                grp.attrs['affinity'] = affinity
                pybel_grp = grp.create_group("pybel")
                processed_grp = pybel_grp.create_group("processed")

                
                ############################### PROCESS THE DOCKING DATA ###############################
                if args.use_docking:
                    # READ THE DOCKING LIGAND POSES

                    # pose_path_list = glob("{}/{}/{}_ligand_pose_*.pdb".format(args.input_docking, name, name)) 
                    pose_path_list = glob("{}/{}/{}_ligand_pose_*.mol2".format(args.input_docking, name, name)) 

                    # if there are poses to read then we will read them, otherwise skip to the crystal structure loop
                    if len(pose_path_list) > 0: 

                        # READ THE DOCKING POCKET DATA
                        
                        #docking_pocket_file = "{}/{}/{}_pocket.mol2".format(args.input_docking, name, name)
                        docking_pocket_file = receptor_path

                        if not os.path.exists(docking_pocket_file):
                            warnings.warn("{} does not exists...this is likely due to failure in chimera preprocessing step, skipping to next complex...".format(docking_pocket_file))
                            # NOTE: not putting a continue here because there may be crystal structure data
                        else:
                    
                            # some docking files are corrupt (have nans for coords) and pybel doesn't do a great job of handling that
                            with open(docking_pocket_file, 'r') as handle:
                                data = handle.read()
                                if "nan" in data:
                                    warnings.warn("{} contains corrupt data, nan's".format(docking_pocket_file))
                                    #continue #TODO: THIS MAY PREVENT THE CRYSTAL STRUCTURE DATA FROM BEING PROCESSED

                                else:                    

                                    pose_pocket_vdw = []
 
                                    try:
                                        #docking_pocket = next(pybel.readfile('pdb', docking_pocket_file))
                                        docking_pocket = next(pybel.readfile('mol2', docking_pocket_file))
                                        pose_pocket_vdw = parse_mol_vdw(mol=docking_pocket, element_dict=element_dict)

                                    except StopIteration:
                                        error = "pybel failed to read {} docking pocket file".format(name)
                                        warnings.warn(error) 
                                        failure_dict["name"].append(name), failure_dict["partition"].append("docking") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                
                                    # in some, albeit strange, cases the pocket consists purely of hydrogen, skip over these if that is the case
                                    if len(pose_pocket_vdw) < 1:
                                        error = "{} docking pocket contains no heavy atoms, unable to store vdw radii".format(name)
                                        warnings.warn(error) 
                                        failure_dict["name"].append(name), failure_dict["partition"].append("docking") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error) 
                    
                                    else:

                                        docking = processed_grp.create_group("docking")
                                        for pose_path in pose_path_list:

                                            try: 
                                                #pose_ligand = next(pybel.readfile('pdb', pose_path))
                                                pose_ligand = next(pybel.readfile('mol2', pose_path))
                                                # do not add the hydrogens! they were already added in chimera and it would reset the charges
                                            except:
                                                error = "no ligand for {} ({} set)".format(name, dataset_name)
                                                warnings.warn(error)
                                                failure_dict["name"].append(name), failure_dict["partition"].append("docking") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                                                continue #TODO:THIS MAY PREVENT THE CRYSTAL STRUCTURE DATA FROM BEING PROCESSED

                                            # extract the van der waals radii for the ligand/pocket
                                            pose_ligand_vdw = parse_mol_vdw(mol=pose_ligand, element_dict=element_dict) 

                                            # in case the ligand consists purely of hydrogen, skip over these if that is the case
                                            if len(pose_ligand_vdw) < 1:
                                                error = "{} ligand consists purely of hydrogen, no heavy atoms to featurize".format(name)
                                                warnings.warn(error) 
                                                failure_dict["name"].append(name), failure_dict["partition"].append("docking") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                                                continue #TODO: THIS MAY PREVENT THE CRYSTAL STRUCTURE DATA FROM BEING PROCESSED
                        
                                            try:
                                                pose_data = featurize_pybel_complex(ligand_mol=pose_ligand, pocket_mol=docking_pocket, name=name, dataset_name=dataset_name)
                                            except RuntimeError as error:
                                                failure_dict["name"].append(name), failure_dict["partition"].append("docking") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                                                continue  #TODO:THIS MAY PREVENT THE CRYSTAL STRUCTURE DATA FROM BEING PROCESSED

                                            pose_ligand_pocket_vdw = np.concatenate([pose_ligand_vdw.reshape(-1), 
                                                                                pose_pocket_vdw.reshape(-1)], axis=0)

                                            # enforce a constraint that the number of atoms for which we have features is equal to number for which we have VDW radii  
                                            assert pose_ligand_pocket_vdw.shape[0] == pose_data.shape[0] 


                                            # CREATE THE DOCKING POSE GROUP
                                            #pose_idx = pose_path.split(".pdb")[0].split("_")[-1]
                                            pose_idx = pose_path.split(".mol2")[0].split("_")[-1]
                                            pose_grp = docking.create_group(pose_idx) 

                                            # Now that we have passed the try/except blocks, featurize and store the docking data 
                                            pose_grp.attrs["van_der_waals"] = pose_ligand_pocket_vdw
                        
                                            pose_dataset = pose_grp.create_dataset("data", data=pose_data, 
                                                                shape=pose_data.shape, dtype='float32', compression='lzf') 

                        
                else: 
                    error = "{} does not contain any pose data".format(name)
                    tqdm.write(error)
                    failure_dict["name"].append(name), failure_dict["partition"].append("docking") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error) 

                
                ############################### PROCESS THE CRYSTAL STRUCTURE DATA ###############################
                
                if args.use_exp: 
                    # BEGIN QUALITY CONTROL: do not create the dataset until data has been verified
                    try:
                        crystal_ligand = next(pybel.readfile('mol2', '%s/%s/%s_ligand.mol2' % (args.input_pdbbind, name, name))) 

                    # do not add the hydrogens! they were already added in chimera and it would reset the charges
                    except:
                        error ="no ligand for {} ({} set)".format(name, dataset_name)
                        warnings.warn(error)
                        failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error) 
                        continue

                    try:
                        crystal_pocket = next(pybel.readfile('mol2', '%s/%s/%s_pocket.mol2' % (args.input_pdbbind, name, name))) 
 
                    except:
                        error = "no pocket for {} ({} set)".format(name, dataset_name)
                        warnings.warn(error)
                        failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                        continue

                    # extract the van der waals radii for the ligand/pocket
                    crystal_ligand_vdw = parse_mol_vdw(mol=crystal_ligand, element_dict=element_dict) 
                
                    # in some, albeit strange, cases the pocket consists purely of hydrogen, skip over these if that is the case
                    if len(crystal_ligand_vdw) < 1:
                        error = "{} ligand consists purely of hydrogen, no heavy atoms to featurize".format(name)
                        warnings.warn(error) 
                        failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                        continue
 
                    crystal_pocket_vdw = parse_mol_vdw(mol=crystal_pocket, element_dict=element_dict)
                    # in some, albeit strange, cases the pocket consists purely of hydrogen, skip over these if that is the case
                    if len(crystal_pocket_vdw) < 1:
                        error = "{} pocket consists purely of hydrogen, no heavy atoms to featurize".format(name)
                        warnings.warn(error) 
                        failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                        continue

                    crystal_ligand_pocket_vdw = np.concatenate([crystal_ligand_vdw.reshape(-1), crystal_pocket_vdw.reshape(-1)], axis=0)
                    try:
                        crystal_data = featurize_pybel_complex(ligand_mol=crystal_ligand, pocket_mol=crystal_pocket, name=name, dataset_name=dataset_name)
                    except RuntimeError as error:
                        failure_dict["name"].append(name), failure_dict["partition"].append("crystal") , failure_dict["set"].append(dataset_name), failure_dict["error"].append(error)
                        continue
                
                    # enforce a constraint that the number of atoms for which we have features is equal to number for which we have VDW radii 
                    assert crystal_ligand_pocket_vdw.shape[0] == crystal_data.shape[0]
    
                    # END QUALITY CONTROL: made it past the try/except blocks....now featurize the data and store into the .hdf file 
                    crystal_grp = processed_grp.create_group("pdbbind")
                    crystal_grp.attrs["van_der_waals"] = crystal_ligand_pocket_vdw 
                    crystal_dataset = crystal_grp.create_dataset("data", data=crystal_data, 
                                                        shape=crystal_data.shape, dtype='float32', compression='lzf') 
                    
      
    failure_df = pd.DataFrame(failure_dict)
    failure_df.to_csv("{}/failure_summary.csv".format(args.output), index=False)

if __name__ == "__main__":
    main()

