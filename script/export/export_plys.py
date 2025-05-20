
import numpy as np
import os
import shutil
from lib.config import cfg
from lib.models.scene import Scene
from plyfile import PlyData, PlyElement
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel


def export_model_plys(path):
    pass

if __name__ == "__main__":
    path = "./data/waymo/training/031"
    export_model_plys(path)
    print("finished")
