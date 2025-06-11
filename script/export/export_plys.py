import numpy as np
import os
import sys
import torch


sys.path.append(os.getcwd())
from lib.config import cfg

cfg.mode = "evaluate"  # not training
import shutil
from lib.models.scene import Scene
from plyfile import PlyData, PlyElement
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.models.gaussian_model_actor import GaussianModelActor
from lib.utils.camera_utils import Camera
from lib.utils.sh_utils import eval_sh

import export_util


def export_model_plys(path):
    frame_id = 50

    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    cameras = train_cameras + test_cameras
    cameras = list(sorted(cameras, key=lambda x: x.id))

    viewpoint_camera: Camera = None
    for camera in cameras:
        if camera.meta["frame_idx"] == frame_id:
            viewpoint_camera = camera
            break

    if viewpoint_camera is None:
        raise ValueError(f"Could not find camera with frame_idx {frame_id}")

    gaussians.set_visibility(list(set(gaussians.model_name_id.keys())))
    gaussians.parse_camera(camera=viewpoint_camera)

    exp_dir = export_util.get_export_dir(path, True)
    gaussians.save_plys(exp_dir)


def calc_colors(t: StreetGaussianModel, camera_center):
    colors = []
    model_names = []
    if t.get_visibility("background"):
        model_names.append("background")
    model_names.extend(t.graph_obj_list)

    for model_name in model_names:
        if model_name == "background":
            continue

        model: GaussianModelActor = getattr(t, model_name)

        max_sh_degree = model.max_sh_degree
        sh_dim = (max_sh_degree + 1) ** 2

        features = model.get_features_fourier(t.frame)
        shs = features.transpose(1, 2).view(-1, 3, sh_dim)

        directions = model.get_xyz - camera_center
        directions = directions / torch.norm(directions, dim=1, keepdim=True)
        sh2rgb = eval_sh(max_sh_degree, shs, directions)
        color = torch.clamp_min(sh2rgb + 0.5, 0.0)
        colors.append(color)

    colors = torch.cat(colors, dim=0)
    return colors


if __name__ == "__main__":
    path = "./data/waymo/training/031"
    export_model_plys(path)
    print("finished")
