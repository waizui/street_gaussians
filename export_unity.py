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
    start_frame = 50
    end_frame = 60
    frames = [start_frame, end_frame]
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
    test_tracklets(frames, res)


def test_tracklets(frames, outputs):
    # [num_frames, max_obj, track_id, x, y, z, qw, qx, qy, qz] ,3-dim
    tkls_w = outputs["obj_tracklets_world"]
    tkls_o = outputs["obj_tracklets"]
    nframes = outputs["num_frames"]
    assert tkls_w.shape == tkls_o.shape

    ego_poses = outputs["ego_poses"]  # [num_frames, 4, 4]

    # obj_tracklets = object_tracklets_vehicle[frames_idx[i]]
    frames_idx = outputs["frames_idx"]

    for frame_idx in frames_idx:
        frame_tkls_w = tkls_w[frame_idx]
        frame_tkls_o = tkls_o[frame_idx]

        trackid_w = frame_tkls_w[:, 0]
        trackid_o = frame_tkls_o[:, 0]

        if trackid_o < 0 or trackid_w < 0:
            continue

        assert np.array_equal(trackid_w, trackid_o)

        pos_rot_w = frame_tkls_w[:, 1:]
        pos_rot_o = frame_tkls_o[:, 1:]

        rotz_mats = (
            general_utils.quaternion_to_matrix(torch.tensor(pos_rot_o[:, -4:]))
            .cpu()
            .numpy()
        )

        ego_pose = ego_poses[frames[0] + frame_idx]  # start-frame based indexing

        for i, rotz_mat in enumerate(rotz_mats):
            obj_pose = np.eye(4)
            obj_pose[:3, :3] = rotz_mat
            obj_pose[:3, 3] = np.array(pos_rot_o[i, :3])
            res_mut = ego_pose @ obj_pose
            assert np.isclose(res_mut[:3, 3], pos_rot_w[i, :3], rtol=1e-5).all()


def export_tracks():
    read_origin_data()


if __name__ == "__main__":
    export_tracks()
