import numpy as np
from pylie import SO3, SE3
from sfm_map import PerspectiveCamera, FeatureTrack, MatchedFrame, MapPoint, Keyframe, KeyPoint, SfmMap
import warnings


def read_dataset(shortest_track_length=3, dataset_path='./data/'):
    with open(dataset_path + 'Holmenkollendatasett.lst') as file:
        img_paths = [dataset_path + line.rstrip() for line in file]

    bundler_file_path = dataset_path + 'Holmenkollendatasett_plane.out'

    return read_bundler_file(bundler_file_path, img_paths, shortest_track_length)


def read_bundler_file(filename, img_paths, shortest_track_length=3, principal_point=np.array([[702, 468]]).T):
    # We will use a camera coordinate system where z points forward, y down.
    pose_bundlerc_c = SE3((SO3(np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])), np.zeros([3, 1])))

    matched_frames = []
    sfm_map = SfmMap()

    with open(filename, 'r') as file:
        line = file.readline()
        if not line.startswith("# Bundle file v0.3"):
            warnings.warn("Warning: Expected v0.3 Bundle file, but first line did not match the convention")

        # Read number of cameras and points.
        num_cameras, num_points = [int(x) for x in next(file).split()]

        # Initialize list of matched frames.
        matched_frames = num_cameras * [None]

        # Read all camera parameters.
        for i in range(num_cameras):
            # Read the intrinsics from the first line.
            f, k1, k2 = [float(x) for x in next(file).split()]

            # We will assume that the data has been undistorted (for simplicity).
            if k1 != 0 or k2 != 0:
                warnings.warn("The current implementation assumes undistorted data. Distortion parameters are ignored")

            # Read the rotation matrix from the next three lines.
            R = np.array([[float(x) for x in next(file).split()] for y in range(3)]).reshape([3, 3])

            # Read the translation from the last line.
            t = np.array([float(x) for x in next(file).split()]).reshape([3, 1])

            # Add matched frame to list.
            f = f * np.ones([2, 1])
            matched_frames[i] = MatchedFrame(i, PerspectiveCamera(f, principal_point), img_paths[i])
            sfm_map.add_keyframe(Keyframe(matched_frames[i], SE3((SO3(R), t)).inverse() @ pose_bundlerc_c))

        # Read the points and matches.
        for i in range(num_points):
            # First line is the position of the point.
            p_w = np.array([float(x) for x in next(file).split()]).reshape([3, 1])

            # Next line is the color.
            color = np.array([int(x) for x in next(file).split()]).reshape([3, 1])

            # The last line is the correspondences in each image where the point is observed.
            view_list = next(file).split()
            num_observed = int(view_list[0])

            # Drop observation if seen from less than shortest_track_length cameras.
            if num_observed < shortest_track_length:
                continue

            # Add observations and detections for each camera the point is observed in.
            curr_track = FeatureTrack()
            curr_map_point = MapPoint(i, p_w)

            for c in range(num_observed):
                view_data = view_list[1 + 4*c: 1 + 4*(c+1)]
                cam_ind = int(view_data[0])
                det_id = int(view_data[1])

                # We have to invert the y-values because we have chosen to let y point downwards.
                det_point = np.array([[float(view_data[2]), -float(view_data[3])]]).T + principal_point

                curr_track.add_observation(matched_frames[cam_ind], det_id)
                matched_frames[cam_ind].add_keypoint(det_id, KeyPoint(det_point, color, curr_track))
                curr_map_point.add_observation(sfm_map.get_keyframe(cam_ind), det_id)

            sfm_map.add_map_point(curr_map_point)

    for frame in matched_frames:
        frame.update_covisible_frames()

    return matched_frames, sfm_map

