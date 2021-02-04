################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# Generate ML-HDF for sgcnn and 3dcnn
################################################################################


# basic
import os
import subprocess
import argparse
import warnings
import numpy as np
import xml.etree.ElementTree as ET
import csv
import h5py
import pandas as pd

# bio-related
import openbabel
import pybel
import rdkit
from rdkit import Chem
from rdkit.Chem import rdchem
from tfbio.data import Featurizer

# multi-processing
import multiprocessing as mp


# decide whether extracting feature from pdb/pdbqt or from mol2
g_convert_mol2 = True


# set up program arguments
parser = argparse.ArgumentParser()
parser.add_argument("--hdfx-path", default="", required=True, help="path to hdfx tool")
parser.add_argument("--input-dir", default="", required=True, help="input complex hdf directory")
parser.add_argument("--input-hdf", default="", help="input complex hdf filename")
parser.add_argument("--output-dir", default="", required=True, help="output directory to store ml-hdf file")
parser.add_argument("--output-suffix", default="_ml.hdf", help="output ml-hdf suffix")
args = parser.parse_args()



def extract_pdbqts(hdfx_path, ligand_hdf_path):
    
    # execute hdfx (don't use os.system)
    cmd = "%s --in %s --nofilename" % (hdfx_path, ligand_hdf_path)
    print(cmd)
    pdbqt_line_list = subprocess.check_output(cmd, shell=True, universal_newlines=True)
    pdbqt_line_list = pdbqt_line_list.splitlines()
    
    ##### temp start #####
    #pdbqt_line_list = []
    #with open("../covid19_misc/worlddrug_old/dock_proc89_hdfx_out.txt", "r") as fp:
    #   pdbqt_line_list = fp.readlines()
    #print(pdbqt_line_list)
    ##### temp end #####

    # separate into each pdbqt data
    head_prefix = "======================="

    pdbqt_info_list = []
    pdbqt_list = []

    new_pdbqt = False
    lig_pdbqt = []
    lig_ind = ''
    lig_name = ''

    line_iter = iter(pdbqt_line_list)
    for line in line_iter:
        if new_pdbqt == True:
            while True:
                if line == None or line.startswith(head_prefix):
                    break
                lig_pdbqt.append(line.rstrip('\n'))
                line = next(line_iter, None)
            pdbqt_list.append(lig_pdbqt)
            pdbqt_info_list.append([lig_ind, lig_name])
            new_pdbqt = False
            lig_pdbqt = []
            lig_ind = ''
            lig_name = ''
            if line == None:
                break

        if line.startswith(head_prefix):
            line = line[len(head_prefix):-len(head_prefix)-1].strip()
            line_sep = line.split('/')
            if line_sep[3] == "status":
                status = int(next(line_iter))
            elif len(line_sep) >= 5 and line_sep[4] == "ligName":
                lig_ind = line_sep[2]
                lig_name = (next(line_iter)).rstrip()
            elif len(line_sep) >= 5 and line_sep[4] == "poses.pdbqt" and status == 1:
                new_pdbqt = True

    return pdbqt_info_list, pdbqt_list


def extract_pdb_poses(pdbqt_data):
    pdb_pose_list = []
    pdb_pose_str = ""
    for line in pdbqt_data:
        if line.startswith("MODEL"):
            pdb_pose_str = ""
            continue

        # Do not write ROOT, ENDROOT, BRANCH, ENDBRANCH, TORSDOF records.
        if line.startswith ('ROOT') or line.startswith ('ENDROOT') \
            or line.startswith ('BRANCH') or line.startswith ('ENDBRANCH') \
            or line.startswith ('TORSDOF'):
            continue

        pdb_pose_str += line.rstrip() + '\n'
        if line.startswith("ENDMDL"):
            pdb_pose_list.append(pdb_pose_str)
            
    return pdb_pose_list


def get_mol_from_pdb(pdb_pose_str, ob_conversion, convert_mol2):
    if convert_mol2:
        mol = openbabel.OBMol()
        ob_conversion.ReadString(mol, pdb_pose_str)
        mol2_pose_str = ob_conversion.WriteString(mol)
        return pybel.readstring('mol2', mol2_pose_str)
    else:
        return pybel.readstring('pdbqt', pdb_pose_str)


def read_element_desc(desc_file):
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


def featurize_pybel_complex(complex_mol, name):

    featurizer = Featurizer()
    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')

    # get complex features
    coords, features = featurizer.get_features(complex_mol, molcode=1)

    #if not (features[:, charge_idx] != 0).any():  # ensures that partial charge on all atoms is non-zero?
    #    raise RuntimeError("invalid charges for the complex {}".format(name))

    # center the coordinates on the complex coordinates
    centroid = coords.mean(axis=0)
    coords -= centroid
    data = np.concatenate((coords, features), axis=1)

    return data


def get_files_ext(a_dir, a_ext):
    return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name)) and name.endswith(a_ext)]


def valid_file(a_path):
    return os.path.isfile(a_path) and os.path.getsize(a_path) > 0


def get_working_file_list(input_dir, input_ext, output_dir, output_suffix, output_summary_suffix):
    input_file_list = get_files_ext(input_dir, input_ext)
    input_file_list.sort()

    working_file_list = []
    for input_fn in input_file_list:
        output_fn = input_fn[:-len(input_ext)-1] + output_suffix
        output_summary_fn = input_fn[:-len(input_ext)-1] + output_suffix.split('.')[0] + output_summary_suffix
        if valid_file(os.path.join(output_dir, output_fn)) and valid_file(os.path.join(output_dir, output_summary_fn)):
            continue
        working_file_list.append((input_fn, output_fn, output_summary_fn))
    return working_file_list


def main():
    # initialize element dict
    element_dict = read_element_desc("elements.xml")

    # initialize failure dict
    failure_dict = {"name": [], "partition": [], "set": [], "error": []}

    # create openbabel conversion instance
    ob_conversion = openbabel.OBConversion()
    ob_conversion.SetInAndOutFormats("pdbqt", "mol2")

    # set up hdfx_path
    hdfx_path = args.hdfx_path
    if len(hdfx_path) == 0:
        hdfx_path = "hdfx"

    # if args.ligand_hdf isn't specified, search the entire input_dir
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
	
    if len(args.input_hdf) == 0:
        working_file_list = get_working_file_list(args.input_dir, "hdf5", args.output_dir, args.output_suffix, "_log.csv")
    else:
        output_fn = args.input_hdf[:-5] + args.output_suffix
        output_summary_fn = args.ligand_hdf[:-5] + (args.output_suffix).split('.')[0] + "_log.csv"
        working_file_list = []
        working_file_list.append([args.input_hdf, output_fn, output_summary_fn])


    # loop over working_file_list
    for (input_fn, output_fn, output_summary_fn) in working_file_list:

        # extract all pdbqt data from hdf
        pdbqt_info_list, pdbqt_list = extract_pdbqts(hdfx_path, os.path.join(args.input_dir, input_fn))
        if len(pdbqt_info_list) == 0:
        	print("no valid pdbqt in %s" % ligand_fn)
	
        # create output ml-hdf
        with h5py.File(os.path.join(args.output_dir, output_fn), 'w') as output_ml_hdf:

            # loop over all pdbqt data
            for pdbqt_ind, (lig_ind, lig_name) in enumerate(pdbqt_info_list):
                print("[%s - %d/%d] extract pdbqt (%s, %s)" % (input_fn, pdbqt_ind+1, len(pdbqt_info_list), lig_ind, lig_name))
            
                # get the current pdbqt data
                pdbqt_data = pdbqt_list[pdbqt_ind]

                # separate into pose data
                pdb_pose_list = extract_pdb_poses(pdbqt_data)
                
                # create a pdb group (a complex with multiple poses)
                dname = "%s_%s" % (lig_ind, lig_name)
                grp = output_ml_hdf.create_group(str(dname))
                grp.attrs['affinity'] = 0
                pybel_grp = grp.create_group("pybel")
                processed_grp = pybel_grp.create_group("processed")
                docking_grp = processed_grp.create_group("docking")

                # loop over poses
                for pdb_pose_ind, pdb_pose_str in enumerate(pdb_pose_list):

                    # get input complex mol instance from pdb pose data (string)
                    complex_mol = get_mol_from_pdb(pdb_pose_str, ob_conversion, g_convert_mol2)

                    # extract feature
                    pose_data = featurize_pybel_complex(complex_mol=complex_mol, name="%s_%s" % (lig_ind, pdb_pose_ind+1))

                    # extract the van der waals radii for the complex
                    vdw = parse_mol_vdw(mol=complex_mol, element_dict=element_dict)
                    assert vdw.shape[0] == pose_data.shape[0]

                    # create a pose group
                    pose_grp = docking_grp.create_group(str(pdb_pose_ind+1))
                    pose_grp.attrs["van_der_waals"] = vdw
                    pose_dataset = pose_grp.create_dataset("data", data=pose_data, shape=pose_data.shape, dtype='float32', compression='lzf')
			
        # save failure summary file
        failure_df = pd.DataFrame(failure_dict)
        failure_df.to_csv(os.path.join(args.output_dir, output_summary_fn), index=False)
	

if __name__ == "__main__":
    main()
