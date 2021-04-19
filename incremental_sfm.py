import cv2
import numpy as np
from pylie import SO3, SE3
from dataset import read_dataset
from sfm_map import SfmMap, Keyframe, MapPoint, MatchedFrame
import open3d as o3d
from optimize import BatchBundleAdjustment, IncrementalBundleAdjustment

"""Assumes undistorted pixel points"""


def initialize_map(frame_0, frame_1):
    # Compute relative pose from two-view geometry.
    kp_0, id_0, kp_1, id_1 = frame_0.extract_correspondences_with_frame(frame_1)
    pose_0_1 = estimate_two_view_relative_pose(frame_0, kp_0, frame_1, kp_1)

    # Triangulate points.
    P_0 = frame_0.camera_model().projection_matrix(SE3())
    P_1 = frame_1.camera_model().projection_matrix(pose_0_1.inverse())
    points_0 = triangulate_points_from_two_views(P_0, kp_0, P_1, kp_1)

    # Add first keyframe as reference frame.
    sfm_map = SfmMap()
    kf_0 = Keyframe(frame_0, SE3())
    sfm_map.add_keyframe(kf_0)

    # Add second keyframe from relative pose.
    kf_1 = Keyframe(frame_1, pose_0_1)
    sfm_map.add_keyframe(kf_1)

    # Add triangulated points as map points relative to reference frame.
    num_matches = len(id_0)
    for i in range(num_matches):
        map_point = MapPoint(i, points_0[:, [i]])
        map_point.add_observation(kf_0, id_0[i])
        map_point.add_observation(kf_1, id_1[i])
        sfm_map.add_map_point(map_point)

    return sfm_map


def estimate_two_view_relative_pose(frame0: MatchedFrame, kp_0: np.ndarray,
                                    frame1: MatchedFrame, kp_1: np.ndarray):
    num_matches = kp_0.shape[1]
    if num_matches < 8:
        return None

    # Compute fundamental matrix from matches.
    F_0_1, _ = cv2.findFundamentalMat(kp_1.T, kp_0.T, cv2.FM_8POINT)

    # Extract the calibration matrices.
    K_0 = frame0.camera_model().calibration_matrix()
    K_1 = frame1.camera_model().calibration_matrix()

    # Compute the essential matrix from the fundamental matrix.
    E_0_1 = K_0.T @ F_0_1 @ K_1

    # Compute the relative pose.
    # Transform detections to normalized image plane (since cv2.recoverPose() only supports common K)
    kp_n_0 = frame0.camera_model().pixel_to_normalised(kp_0)
    kp_n_1 = frame1.camera_model().pixel_to_normalised(kp_1)
    K_n = np.identity(3)
    _, R_0_1, t_0_1, _ = cv2.recoverPose(E_0_1, kp_n_1.T, kp_n_0.T, K_n)

    return SE3((SO3(R_0_1), t_0_1))


def triangulate_points_from_two_views(P_0: MatchedFrame, kp_0: np.ndarray,
                                      P_1: MatchedFrame, kp_1: np.ndarray):
    # Triangulate wrt frame 0.
    points_hom = cv2.triangulatePoints(P_0, P_1, kp_0, kp_1)
    return points_hom[:-1, :] / points_hom[-1, :]


def track_map(sfm_map, frame_new):
    # Find correspondences with map.
    kp, points_0, frame_map_corr = sfm_map.extract_2d_3d_correspondences(frame_new)

    # Estimate initial pose wrt map with PnP.
    pose_w_new = estimate_pose_from_map_correspondences(frame_new, kp, points_0)
    return frame_map_corr, pose_w_new


def estimate_pose_from_map_correspondences(frame: MatchedFrame, kp: np.ndarray, points_w: np.ndarray):
    # Estimate initial pose with a (new) PnP-method.
    K = frame.camera_model().calibration_matrix()
    _, theta_vec, t = cv2.solvePnP(points_w.T, kp.T, K, None, flags=cv2.SOLVEPNP_SQPNP)
    pose_c_w = SE3((SO3.Exp(theta_vec), t.reshape(3, 1)))

    return pose_c_w.inverse()


def add_as_keyframe_to_map(sfm_map, frame_new, pose_w_new, frame_map_corr):
    # Add new keyframe
    kf_new = Keyframe(frame_new, pose_w_new)
    sfm_map.add_keyframe(kf_new)

    # Add map point observations to new keyframe.
    for kp_id, map_point in frame_map_corr.items():
        map_point.add_observation(kf_new, kp_id)
    return kf_new


def find_and_add_new_map_points(sfm_map, kf_new):
    # Find new correspondences with the keyframes that are not map points.
    corr_for_keyframes = sfm_map.extract_correspondences_for_new_map_points(kf_new)

    # Triangulate new points and add new map points
    keyframe_ids = sfm_map.get_keyframe_ids()
    for kf_old, (kp_old, kp_new, tracks) in corr_for_keyframes.items():
        P_old = kf_old.camera_model().projection_matrix(kf_old.pose_w_c().inverse())
        P_new = kf_new.camera_model().projection_matrix(kf_new.pose_w_c().inverse())
        points_w = triangulate_points_from_two_views(P_old, kp_old, P_new, kp_new)

        # TODO: Find a better solution for assigning ids to MapPoints!
        start_ind = len(sfm_map.get_map_points())
        for i in range(len(tracks)):
            map_point = MapPoint(start_ind + i, points_w[:, [i]])

            for frame, kp_id in tracks[i].get_observations().items():
                if frame.id() in keyframe_ids:
                    map_point.add_observation(sfm_map.get_keyframe(frame.id()), kp_id)

            sfm_map.add_map_point(map_point)


def interactive_isfm():
    matched_frames, _ = read_dataset(shortest_track_length=3)

    # Choose optimization method, BatchBundleAdjustment or IncrementalBundleAdjustment.
    optimizer = BatchBundleAdjustment()

    # Choose the two first frames for initialization
    frame_0 = matched_frames[0]
    frame_1 = matched_frames[1]

    # Initialize map from two-view geometry.
    sfm_map = initialize_map(frame_0, frame_1)

    # You can here choose which images to add to the map in add_new_frame().
    next_frames = matched_frames[2::]

    # Callback for optimizing the map (press 'O')
    def optimize(vis):
        # Apply BA.
        optimizer.full_bundle_adjustment_update(sfm_map)

        vis.clear_geometries()
        for geom in get_geometry():
            vis.add_geometry(geom, reset_bounding_box=False)

    # Callback for adding new frame to the map (press 'A')
    def add_new_frame(vis):
        if not next_frames:
            return

        # Get next frame
        frame_new = next_frames.pop(0)
        print("Adding frame " + str(frame_new.id()))

        # Find 2d-3d correspondences with map and compute initial pose with respect to the map.
        frame_map_corr, pose_w_new = track_map(sfm_map, frame_new)

        # Insert frame as keyframe into the map
        kf_new = add_as_keyframe_to_map(sfm_map, frame_new, pose_w_new, frame_map_corr)

        # Find new correspondences, triangulate and add as map points.
        find_and_add_new_map_points(sfm_map, kf_new)

        vis.clear_geometries()
        for geom in get_geometry():
            vis.add_geometry(geom, reset_bounding_box=False)

    # Helper function for extracting the visualization elements from the map.
    def get_geometry():
        poses = sfm_map.get_keyframe_poses()
        p, c = sfm_map.get_pointcloud()

        axes = []
        for pose in poses:
            axes.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0).transform(pose.to_matrix()))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(p.T)
        pcd.colors = o3d.utility.Vector3dVector(c.T / 255)

        return [pcd] + axes

    # Create visualizer.
    key_to_callback = {}
    key_to_callback[ord("O")] = optimize
    key_to_callback[ord("A")] = add_new_frame
    o3d.visualization.draw_geometries_with_key_callbacks(get_geometry(), key_to_callback)


if __name__ == '__main__':
    interactive_isfm()
