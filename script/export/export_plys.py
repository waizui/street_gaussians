import numpy as np
import os
import sys
import torch

sys.path.append(os.getcwd())

import export_util


def export_model_plys(frame_id, path):
    gaussians, _ = export_util.load_model(frame_id)
    exp_dir = export_util.get_export_dir(path, True)
    gaussians.save_plys(exp_dir)


if __name__ == "__main__":
    path = "./data/waymo/training/031"
    frame_id = 50
    export_model_plys(frame_id, path)
    print("finished")
