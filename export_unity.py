from lib.utils import waymo_utils
from lib.utils import general_utils
import numpy as np
import torch


class Transform:
    pos_rot: list

    def __init__(self, pos_rot) -> None:
        self.pos_rot = pos_rot


class Trajectory:
    id_obj: int
    transform: list

    def __init__(self, id) -> None:
        self.transform = []
        self.id_obj = id


def read_origin_data(path, frames, cameras=[0, 1, 2]):
    """
    result['num_frames'] = num_frames
    result['exts'] = exts
    result['ixts'] = ixts
    result['poses'] = poses
    result['c2ws'] = c2ws
    result['ego_poses'] = ego_frame_poses
    result['obj_tracklets_world'] = obj_tracklets_world # np.matmul(ego_pose, obj_pose_vehicle)
    result['obj_tracklets'] = object_tracklets_vehicle
    result['obj_info'] = object_info
    result['frames'] = frames
    result['cams'] = cams
    result['frames_idx'] = frames_idx
    result['image_filenames'] = image_filenames
    result['cams_timestamps'] = cams_timestamps
    result['tracklet_timestamps'] = frames_timestamps
    """
    return waymo_utils.generate_dataparser_outputs(path, frames, False, cameras=cameras)


def export_tracks():
    path = "./data/waymo/training/031"
    frames = [1, 120]
    data_org = read_origin_data(path, frames)

    frames_idx = data_org["frames_idx"]

    tkls_w = data_org["obj_tracklets_world"]
    max_obj = tkls_w.shape[1]
    trajectories = [Trajectory(i) for i in range(max_obj)]

    for frame_idx in frames_idx:
        frame_tkls_w = tkls_w[frame_idx]
        for obj in range(max_obj):
            frame_tkl_w = frame_tkls_w[obj]
            if frame_tkl_w[0] < 0:
                break
            t = Transform(frame_tkl_w[1:])
            trajectories[obj].transform.append(t)
    return trajectories


if __name__ == "__main__":
    export_tracks()
