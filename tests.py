from export_unity import *


def test_tracklets():
    path = "./data/waymo/training/031"
    frames = [50, 60]
    data_org = read_origin_data(path, frames)

    # [num_frames, max_obj, track_id, x, y, z, qw, qx, qy, qz] ,3-dim
    tkls_w = data_org["obj_tracklets_world"]
    tkls_o = data_org["obj_tracklets"]
    nframes = data_org["num_frames"]
    assert tkls_w.shape == tkls_o.shape

    ego_poses = data_org["ego_poses"]  # [num_frames, 4, 4]

    # obj_tracklets = object_tracklets_vehicle[frames_idx[i]]
    frames_idx = data_org["frames_idx"]

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
        nitem = rotz_mats.shape[0]
        obj_pose = np.zeros(nitem * 4 * 4).reshape([nitem, 4, 4])
        obj_pose[:, :, :] = np.eye(4)
        obj_pose[:, :3, :3] = rotz_mats
        obj_pose[:, :3, 3] = np.array(pos_rot_o[:, :3])
        res_mut = ego_pose @ obj_pose
        assert np.isclose(res_mut[:, :3, 3], pos_rot_w[:, :3], rtol=1e-5).all()
    print("test passed")


if __name__ == "__main__":
    test_tracklets()
