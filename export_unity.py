from lib.utils import waymo_utils
from lib.utils import system_utils
import numpy as np
import os


class Trajectory:
    trackdata: list  # [track_id, x, y, z, qw, qx, qy, qz]
    obj_id: int

    def __init__(self, id) -> None:
        self.obj_id = id
        self.trackdata = []


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


def read_trajectory(path, frames):
    data_org = read_origin_data(path, frames)

    frames_idx = data_org["frames_idx"]

    tkls_w = data_org["obj_tracklets_world"]
    max_obj = tkls_w.shape[1]
    traj_arr = [Trajectory(i) for i in range(max_obj)]

    for frame_idx in frames_idx:
        frame_tkls_w = tkls_w[frame_idx]
        for obj in range(max_obj):
            frame_tkl_w = frame_tkls_w[obj]
            if frame_tkl_w[0] < 0:
                break
            traj_arr[obj].trackdata.append(frame_tkl_w)
    return traj_arr


def export_trajectory(path, frames):
    """
    [obj_id, track_id, x, y, z, qw, qx, qy, qz]
    """
    exp_path = "./output/" + os.path.relpath(path, os.getcwd()) + "/trajectories.csv"
    system_utils.mkdir_p(os.path.dirname(exp_path))
    trajs = read_trajectory(path, frames)

    with open(exp_path, "w") as f:
        lines = []
        for tj in trajs:
            for tr in tj.trackdata:
                line = str(tj.obj_id) + "," + ",".join(tr.flatten().astype(str)) + "\n"
                lines.append(line)
        f.writelines(lines)


if __name__ == "__main__":
    path = "./data/waymo/training/031"
    frames = [1, 120]
    export_trajectory(path, frames)
    print("finished")
