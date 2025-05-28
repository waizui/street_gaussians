import os
import sys

sys.path.append(os.getcwd())
from lib.utils import waymo_utils
import numpy as np
import export_util


class Trajectory:
    id: int

    def __init__(self, id) -> None:
        self.id = id
        # (num_frames,track_id) , [track_id, num_frames , x, y, z, qw, qx, qy, qz]
        self.dict_trackdata = {}

    def get_trackdata(self):
        return self.dict_trackdata.values()

    def append(self, data):
        key = (data[0], data[1])
        if key in self.dict_trackdata:
            return
        self.dict_trackdata[key] = data


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
    stamps = data_org["tracklet_timestamps"]

    # [num_frames, max_obj, len[track_id, x, y, z, qw, qx, qy, qz]]
    tkls_w = data_org["obj_tracklets_world"]
    max_obj = tkls_w.shape[1]
    traj_arr = {}

    for cam_id, frame_idx in enumerate(frames_idx):
        for obj in range(max_obj):
            frame_tkl_w = tkls_w[frame_idx][obj]
            if frame_tkl_w[0] < 0:  # -1 no data
                break

            pos = frame_tkl_w[1:4]
            y_up_mat = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            frame_tkl_w[1:4] = y_up_mat @ np.array(pos)

            track_id = frame_tkl_w[0]
            if (track_id) not in traj_arr:
                traj_arr[track_id] = Trajectory(track_id)

            data = np.insert(frame_tkl_w, 1, frame_idx)
            data = np.append(data,stamps[frame_idx])
            traj_arr[track_id].append(data)
    return traj_arr.values()


def export_trajectory(path, frames):
    """
    [track_id, frame_id, x, y, z, qw, qx, qy, qz]
    """
    exp_path = export_util.get_export_path(
        path, f"trajectories_frames_{frames[0]}_{frames[1]}.csv"
    )

    trajs = read_trajectory(path, frames)

    with open(exp_path, "w") as f:
        lines = []
        for tj in trajs:
            for tr in tj.get_trackdata():
                line = ",".join(tr.flatten().astype(str)) + "\n"
                lines.append(line)
        f.writelines(lines)
    print(f"trajectory written to {exp_path}")


if __name__ == "__main__":
    path = "./data/waymo/training/031"
    frames = [0, 90]
    export_trajectory(path, frames)
    print("finished")
