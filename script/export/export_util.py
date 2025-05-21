import os
import sys

sys.path.append(os.getcwd())
from lib.utils import system_utils


def get_export_dir(relative_path, create):
    rel = os.path.relpath(relative_path, os.getcwd())
    exp_dir = os.path.join("./output/", rel)
    if create:
        system_utils.mkdir_p(exp_dir)
    return exp_dir


def get_export_path(relative_path, filename, create_dir=True):
    path = get_export_dir(relative_path, create_dir)
    return os.path.join(path, filename)
