#!/usr/bin/bash
################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# based on implementation provide in https://gitlab.com/cheminfIBB/pafnucy/blob/master/pdbbind_data.ipynb
################################################################################

# Prepare pockets with UCSF Chimera - pybel sometimes fails to calculate the charges.
# Even if Chimera fails to calculate several charges (mostly for non-standard residues),
# it returns charges for other residues.

path=$@


# get list of pdb files from stdin and iterate over them. each instance of this script appends
# its PID to the tmp.mol2 file in order to prevent race conditions, enabling this to be run with
# gnu parallel


tmp_file=$$_tmp.mol2

echo "my tmp file is ${tmp_file}"

for pdbfile in $path; do

        echo ${pdbfile}
        mol2file=${pdbfile%pdb}mol2

        # NOTICED THAT SOME INPUTS seem to never finish chimera step
        echo -e "open $pdbfile \n addh \n addcharge \n write format mol2 0 $$_tmp.mol2 \n stop" | chimera --nogui 
        # Do not use TIP3P atom types, pybel cannot read them
        sed 's/H\.t3p/H    /' ${tmp_file} | sed 's/O\.t3p/O\.3  /' > $mol2file

done
echo "finished processing"
