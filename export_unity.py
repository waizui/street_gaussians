from lib.utils import waymo_utils
from lib.utils import general_utils
import numpy as np
import torch


class Transform:
    pos: list
    trans: list
    rot: list
    mat_o2w: list


class Trajectory:
    obj_id: int

    def __init__(self) -> None:
        self.transform = []
        self.obj_id = -1


def read_origin_data():
    path = "./data/waymo/training/031"
    frames = [50, 60]
    cameras = [0, 1, 2]

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

    res = waymo_utils.generate_dataparser_outputs(path, frames, False, cameras=cameras)
    test_tracklets(res)


def test_tracklets(outputs):
    # [num_frames, max_obj, track_id, x, y, z, qw, qx, qy, qz] ,3-dim
    tr_w = outputs["obj_tracklets_world"]
    tr_o = outputs["obj_tracklets"]
    assert tr_w.shape == tr_o.shape

    ego_poses = outputs["ego_poses"]  # [num_frames, 4, 4]

    nframes = outputs["num_frames"]
    for frame_id in range(nframes):
        obj_ids = tr_w[frame_id, :, 0] > 0

        for obj_idx in np.where(obj_ids)[0]:
            trackid = tr_w[frame_id, obj_idx, 0]
            pos_rot_w = tr_w[frame_id, obj_idx, 1:].flatten()
            pos_rot_o = tr_o[frame_id, obj_idx, 1:].flatten()

            rotz_mat = (
                general_utils.quaternion_to_matrix(torch.tensor(pos_rot_o[None, -4:]))
                .reshape((3, 3))
                .cpu()
                .numpy()
            )

            obj_pose = np.eye(4)
            obj_pose[:3, :3] = rotz_mat
            obj_pose[:3, 3] = np.array(pos_rot_o[:3])

            ego = ego_poses[frame_id]
            res_mut = ego @ obj_pose

            print(res_mut[:3, 3])
            print("-" * 20)
            print(pos_rot_w[:3])
            assert np.isclose(res_mut[:3, 3], pos_rot_w[:3], rtol=1e-5).all()


def export_tracks():
    read_origin_data()


if __name__ == "__main__":
    export_tracks()
