################################################################################
# Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the LICENSE file for details.
# SPDX-License-Identifier: MIT
#
# Fusion models for Atomic and molecular STructures (FAST)
# File utility functions
################################################################################


import os


def get_files(a_dir):
	return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name))]

def get_files_prefix(a_dir, a_prefix):
	return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name)) and name.startswith(a_prefix)]

def get_files_ext(a_dir, a_ext):
	return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name)) and name.endswith(a_ext)]

def get_files_prefix_ext(a_dir, a_prefix, a_ext):
	return [name for name in os.listdir(a_dir) if os.path.isfile(os.path.join(a_dir, name)) and name.startswith(a_prefix) and name.endswith(a_ext)]

def get_subdirectories(a_dir):
	return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_subdirectories_prefix(a_dir, a_prefix):
	return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name)) and name.startswith(a_prefix)]

def valid_file(a_path):
	return os.path.isfile(a_path) and os.path.getsize(a_path) > 0
