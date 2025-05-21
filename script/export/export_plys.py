import numpy as np
import os
import sys

sys.path.append(os.getcwd())
import shutil
from lib.config import cfg
from lib.models.scene import Scene
from plyfile import PlyData, PlyElement
from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
import export_util


def export_model_plys(path):
    inverse_opacity = lambda x: np.log(x / (1 - x))
    inverse_scale = lambda x: np.log(x)

    frame_id = 50

    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    cameras = train_cameras + test_cameras
    cameras = list(sorted(cameras, key=lambda x: x.id))

    viewpoint_camera = None
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


if __name__ == "__main__":
    path = "./data/waymo/training/031"
    export_model_plys(path)
    print("finished")
