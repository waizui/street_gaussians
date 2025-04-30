#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from errno import EEXIST
from os import makedirs, path
import os
import shutil

from torchvision.utils import pathlib

def mkdir_p(folder_path):
    # Creates a directory. equivalent to using mkdir -p on the command line
    try:
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            raise

def searchForMaxIteration(folder):
    saved_iters = [int(fname.split('.')[0].split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)

def clear_dir(folder_path):
    dir_path = pathlib.Path(folder_path)
    if not dir_path.is_dir():
        return
    for item_path in dir_path.iterdir():
        try:
            if item_path.is_file() or item_path.is_symlink():
                item_path.unlink()  
            elif item_path.is_dir():
                shutil.rmtree(item_path)  
        except Exception as e:
            print(f"Error removing {item_path}: {e}")

def del_dir(folder_path):
    dir_path = pathlib.Path(folder_path)
    if dir_path.exists():
        if dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
            except OSError as e:
                print(f"Error removing directory {dir_path}: {e}")
