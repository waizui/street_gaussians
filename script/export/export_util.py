import os
import sys

sys.path.append(os.getcwd())

from lib.config import cfg
cfg.mode = "evaluate"  # not training

from lib.utils import system_utils

from lib.models.scene import Scene
from lib.datasets.dataset import Dataset
from lib.models.street_gaussian_model import StreetGaussianModel
from lib.utils.camera_utils import Camera
from typing import Tuple, Union


def get_export_dir(relative_path, create):
    rel = os.path.relpath(relative_path, os.getcwd())
    exp_dir = os.path.join("./output/", rel)
    if create:
        system_utils.mkdir_p(exp_dir)
    return exp_dir


def get_export_path(relative_path, filename, create_dir=True):
    path = get_export_dir(relative_path, create_dir)
    return os.path.join(path, filename)


def load_model(frame_id) -> Tuple[StreetGaussianModel, Camera]:
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    train_cameras = scene.getTrainCameras()
    test_cameras = scene.getTestCameras()
    cameras = train_cameras + test_cameras
    cameras = list(sorted(cameras, key=lambda x: x.id))

    viewpoint_camera: Union[Camera, None] = None
    for camera in cameras:
        if camera.meta["frame_idx"] == frame_id:
            viewpoint_camera = camera
            break

    if viewpoint_camera is None:
        raise ValueError(f"Could not find camera with frame_idx {frame_id}")

    gaussians.set_visibility(list(set(gaussians.model_name_id.keys())))
    gaussians.parse_camera(camera=viewpoint_camera)

    return gaussians, viewpoint_camera
