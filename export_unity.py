from lib.utils import waymo_utils


class Transform:
    translation: list
    rotation: list


class Trajectory:
    obj_id: int
    transform: Transform


def read_tracks_data():
    path = "./data/waymo/training/031"
    frames = [50, 60]
    cameras = [0, 1, 2]

    """
    result['num_frames'] = num_frames
    result['exts'] = exts
    result['ixts'] = ixts
    result['poses'] = poses
    result['c2ws'] = c2ws
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
    tks = res["obj_tracklets"]


def export_tracks():
    read_tracks_data()


if __name__ == "__main__":
    export_tracks()
