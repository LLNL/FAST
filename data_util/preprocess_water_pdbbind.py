################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# This is a script that will parse a directory containing pdbbind data (.pdb files) and remove the water molecules from the data
################################################################################


import multiprocessing as mp
import os
from rdkit import Chem
from io import StringIO
import pdbfixer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, default="$DATA/v2007_amber_with_sdf" , help="input path to raw pbdbbind directory")
parser.add_argument("--output", required=True, default="$DATA/v2007_no_water_no_polar_hydrogen", help="output path for processed data directory")

args = parser.parse_args()

from Bio.PDB import PDBParser, PDBIO, Select
from xpdb import SloppyStructureBuilder
from glob import glob
import tqdm

import simtk


class WaterSelect(Select):
    def accept_residue(self, residue):
        if residue.get_resname().lower() == 'wat':
            return 0
        else:
            return 1


def process_pdb(pdb_file):
    # use the pdbfixer utility to make sure the pdbfile is properly represented

    out_file = args.output + str(pdb_file).replace(args.input, "")
    out_dir = out_file.replace("/com.pdb", "")

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(pdb_file) as f:

        fixer = pdbfixer.PDBFixer(pdbfile=f)
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)

        with open(out_file, 'w') as handle:

            simtk.openmm.app.PDBFile.writeFile(fixer.topology, fixer.positions, handle)

        # now read back using biopython :) and apply the water/hydrogen filters
        parser = PDBParser(QUIET=True, structure_builder=SloppyStructureBuilder())
        structure = parser.get_structure('', out_file)

        io = PDBIO()
        io.set_structure(structure)

        io.save(out_file, WaterSelect())


def main():

    pdb_files = glob(args.input+"/*/com.pdb")
    print("found {} pdb files...".format(len(pdb_files)))

    with mp.Pool(64) as pool:
        list(tqdm.tqdm(pool.imap_unordered(process_pdb, pdb_files), total=len(pdb_files)))


if __name__ == "__main__":
    main()
