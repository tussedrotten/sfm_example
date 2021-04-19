import numpy as np
import cv2

from dataclasses import dataclass
from pylie import SE3, SO3


class PerspectiveCamera:
    """Camera model for the perspective camera"""

    def __init__(self, f: np.ndarray, c: np.ndarray):
        self._f = f
        self._c = c

    @staticmethod
    def looks_at_pose(camera_pos_w: np.ndarray, target_pos_w: np.ndarray, up_vector_w: np.ndarray):
        cam_to_target_w = target_pos_w - camera_pos_w
        cam_z_w = cam_to_target_w.flatten() / np.linalg.norm(cam_to_target_w)

        cam_to_right_w = np.cross(-up_vector_w.flatten(), cam_z_w)
        cam_x_w = cam_to_right_w / np.linalg.norm(cam_to_target_w)

        cam_y_w = np.cross(cam_z_w, cam_x_w)

        return SE3((SO3(np.vstack((cam_x_w, cam_y_w, cam_z_w)).T), camera_pos_w))

    @staticmethod
    def project_to_normalised_3d(x_c: np.ndarray):
        return x_c / x_c[-1]

    @classmethod
    def project_to_normalised(cls, x_c: np.ndarray):
        xn = cls.project_to_normalised_3d(x_c)
        return xn[:2]

    def fx(self):
        return self._f[0].item()

    def fy(self):
        return self._f[1].item()

    def cx(self):
        return self._c[0].item()

    def cy(self):
        return self._c[1].item()

    def calibration_matrix(self):
        K = np.identity(3)
        np.fill_diagonal(K[:2, :2], self._f)
        K[:2, [2]] = self._c
        return K

    def projection_matrix(self, pose_c_w: SE3):
        return self.calibration_matrix() @ pose_c_w.to_matrix()[:3, :]

    def project_to_pixel(self, x_c: np.ndarray):
        return (self.project_to_normalised(x_c) * self._f) + self._c

    def pixel_to_normalised(self, u: np.ndarray):
        return (u - self._c) / self._f

    def pixel_cov_to_normalised_cov(self, pixel_cov: np.ndarray):
        S_f = np.diag(1 / self._f.flatten())
        return S_f @ pixel_cov @ S_f.T

    @classmethod
    def reprojection_error_normalised(cls, x_c: np.ndarray, measured_x_n: np.ndarray):
        return measured_x_n[:2] - cls.project_to_normalised(x_c)

    def reprojection_error_pixel(self, x_c: np.ndarray, measured_u: np.ndarray):
        return measured_u - self.project_to_pixel(x_c)


class FeatureTrack:
    def __init__(self):
        self._observations = {}

    def add_observation(self, frame, keypoint_id):
        self._observations[frame] = keypoint_id

    def get_observing_frames(self):
        return self._observations.keys()

    def get_observations(self):
        return self._observations


class KeyPoint:
    def __init__(self, point, color, track=None):
        self._point = point
        self._color = color
        self._track = track

    def point(self):
        return self._point

    def color(self):
        return self._color

    def track(self):
        return self._track

    def set_track(self, track):
        self._track = track

    def has_track(self):
        return self._track is not None


class MatchedFrame:
    def __init__(self, id, camera_model, img_path):
        self._id = id
        self._camera_model = camera_model
        self._img_path = img_path
        self._keypoints = {}
        self._frame_matches = {}

    def id(self):
        return self._id

    def load_image(self):
        return cv2.cvtColor(cv2.imread(self._img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB)

    def add_keypoint(self, keypoint_id, keypoint):
        self._keypoints[keypoint_id] = keypoint

    def get_keypoint(self, keypoint_id):
        return self._keypoints[keypoint_id]

    def keypoints(self):
        return self._keypoints

    def number_of_keypoints(self):
        return len(self._keypoints)

    def add_track(self, keypoint_id, track):
        self._keypoints[keypoint_id].set_track(track)

    def covisible_frames(self):
        return self._frame_matches.keys()

    def covisible_frame_ids(self):
        return [frame.id() for frame in self._frame_matches.keys() if frame != self]

    def find_covisible_frames(self, min_number_of_matches=1):
        return [frame for frame, num_matches in self._frame_matches.items() if num_matches >= min_number_of_matches]

    def update_covisible_frames(self):
        self._frame_matches = {}

        for id, keypoint in self._keypoints.items():
            if not keypoint.has_track():
                continue

            observers = keypoint.track().get_observing_frames()

            for obs in observers:
                if obs in self._frame_matches.keys():
                    self._frame_matches[obs] += 1
                    continue

                self._frame_matches[obs] = 1

    def extract_correspondences_with_frame(self, other_frame):
        num_matches = self.number_of_matches(other_frame)
        if num_matches == 0:
            return [], []

        kp_0 = np.zeros([2, num_matches])
        id_0 = [int] * num_matches
        kp_1 = np.zeros([2, num_matches])
        id_1 = [int] * num_matches

        curr_match = 0
        for id, keypoint in self._keypoints.items():
            if other_frame in keypoint.track().get_observing_frames():
                kp_0[:, [curr_match]] = keypoint.point()
                id_0[curr_match] = id

                other_id = keypoint.track().get_observations()[other_frame]
                kp_1[:, [curr_match]] = other_frame.get_keypoint(other_id).point()
                id_1[curr_match] = other_id
                curr_match += 1

        return kp_0, id_0, kp_1, id_1

    def number_of_matches(self, other_frame):
        if other_frame not in self._frame_matches:
            return 0

        return self._frame_matches[other_frame]

    def camera_model(self):
        return self._camera_model


class Keyframe:
    def __init__(self, frame: MatchedFrame, pose_w_c: SE3):
        self._id = frame.id()
        self._frame = frame
        self._pose_w_c = pose_w_c
        self._map_points = {}
        self._has_been_optimized = False

    def id(self):
        return self._id

    def frame(self):
        return self._frame

    def get_covisible_frames(self):
        return self.get_covisible_frames_by_weight(1)

    def get_covisible_frames_by_weight(self, weight):
        return self._frame.find_covisible_frames(weight)

    def add_map_point(self, keypoint_id, map_point):
        self._map_points[keypoint_id] = map_point

    def get_map_point(self, keypoint_id):
        return self._map_points[keypoint_id]

    def find_corresponding_map_points(self, frame: MatchedFrame):
        _, kpid_kf, _, kpid_frame = self._frame.extract_correspondences_with_frame(frame)

        corr = {}
        for i in range(len(kpid_kf)):
            if kpid_kf[i] in self._map_points.keys():
                corr[kpid_frame[i]] = self._map_points[kpid_kf[i]]

        return corr

    def get_keypoint(self, keypoint_id):
        return self._frame.get_keypoint(keypoint_id)

    def get_observations(self):
        return self._map_points

    def pose_w_c(self):
        return self._pose_w_c

    def update_pose_w_c(self, updated_pose_w_c):
        self._pose_w_c = updated_pose_w_c
        self._has_been_optimized = True

    def has_been_optimized(self):
        return self._has_been_optimized

    def camera_model(self):
        return self._frame.camera_model()

    def find_keypoint_tracks_without_map_points(self):
        tracks = {kp_id: kp.track() for kp_id, kp in self._frame.keypoints().items()
                  if kp_id not in self._map_points.keys()}
        return tracks


class MapPoint:
    def __init__(self, id, point_w):
        self._id = id
        self._point_w = point_w
        self._color = None
        self._observations = {}
        self._has_been_optimized = False

    def id(self):
        return self._id

    def point_w(self):
        return self._point_w

    def update_point_w(self, updated_point_w):
        self._point_w = updated_point_w
        self._has_been_optimized = True

    def has_been_optimized(self):
        return self._has_been_optimized

    def color(self):
        return self._color

    def add_observation(self, keyframe, keypoint_id):
        self._observations[keyframe] = keypoint_id
        keyframe.add_map_point(keypoint_id, self)

        # For simplicity, use first observation's color (since they here are the same anyway)
        if self._color is None:
            self._color = keyframe.get_keypoint(keypoint_id).color()

    def get_observing_keyframes(self):
        return self._observations.keys()

    def get_observations(self):
        return self._observations


class SfmMap:
    def __init__(self):
        self._keyframes = {}
        self._map_points = {}
        self._newly_added_keyframes = []
        self._newly_added_map_points = []

    def add_keyframe(self, keyframe):
        self._keyframes[keyframe.id()] = keyframe
        self._newly_added_keyframes.append(keyframe)

    def add_map_point(self, map_point):
        self._map_points[map_point.id()] = map_point
        self._newly_added_map_points.append(map_point)

    def has_been_updated(self):
        self._newly_added_keyframes = [kf for kf in self._newly_added_keyframes if not kf.has_been_optimized()]
        self._newly_added_map_points = [mp for mp in self._newly_added_map_points if not mp.has_been_optimized()]

    def get_new_keyframes_not_optimized(self):
        return [kf for kf in self._newly_added_keyframes if not kf.has_been_optimized()]

    def get_new_map_points_not_optimized(self):
        return [mp for mp in self._newly_added_map_points if not mp.has_been_optimized()]

    def number_of_keyframes(self):
        return len(self._keyframes)

    def number_of_map_points(self):
        return len(self._map_points)

    def get_keyframe(self, keyframe_id):
        return self._keyframes[keyframe_id]

    def get_keyframes(self):
        return self._keyframes.values()

    def get_keyframe_ids(self):
        return self._keyframes.keys()

    def get_map_point(self, map_point_id):
        return self._map_points[map_point_id]

    def get_map_points(self):
        return self._map_points.values()

    def get_keyframe_poses(self):
        return [keyframe.pose_w_c() for _, keyframe in self._keyframes.items()]

    def get_pointcloud(self):
        num_points = self.number_of_map_points()
        points = np.zeros([3, num_points])
        colors = np.zeros([3, num_points])

        for i, map_point in enumerate(self._map_points.values()):
            points[:, [i]] = map_point.point_w()
            colors[:, [i]] = map_point.color()

        return points, colors

    def find_covisible_keyframes(self, frame: MatchedFrame):
        covisible_frame_ids = frame.covisible_frame_ids()
        return [keyframe for frame_id, keyframe in self._keyframes.items() if frame_id in covisible_frame_ids]

    def find_correspondences_with_map(self, frame: MatchedFrame):
        keyframes = self.find_covisible_keyframes(frame)

        frame_map_corr = {}
        for keyframe in keyframes:
            # This ensures max one match for each keypoint in frame.
            # Could be more efficient, since it overwrites common points between keyframes, but WTH!
            frame_map_corr.update(keyframe.find_corresponding_map_points(frame))

        return frame_map_corr

    def extract_2d_3d_correspondences(self, frame: MatchedFrame):
        frame_map_corr = self.find_correspondences_with_map(frame)

        num_matches = len(frame_map_corr)
        kp = np.zeros([2, num_matches])
        points_w = np.zeros([3, num_matches])

        for i, (frame_kp_id, map_point) in enumerate(frame_map_corr.items()):
            kp[:, [i]] = frame.get_keypoint(frame_kp_id).point()
            points_w[:, [i]] = map_point.point_w()

        return kp, points_w, frame_map_corr

    def extract_correspondences_for_new_map_points(self, new_keyframe: Keyframe):
        candidate_tracks = new_keyframe.find_keypoint_tracks_without_map_points()
        covisible_keyframes = self.find_covisible_keyframes(new_keyframe.frame())

        # Sort according to how far away the other keyframes are from the new.
        # We do this to ensure the longest possible baseline.
        covisible_keyframes.sort(key=lambda kf: np.linalg.norm(
            kf.pose_w_c().translation - new_keyframe.pose_w_c().translation), reverse=True)

        kf_matches = {keyframe: {} for keyframe in covisible_keyframes}
        for kp_id, track in candidate_tracks.items():
            for keyframe in covisible_keyframes:
                curr_kf_frame = keyframe.frame()
                if curr_kf_frame in track.get_observing_frames():
                    curr_kf_frame_kp_id = track.get_observations()[curr_kf_frame]
                    kf_matches[keyframe][kp_id] = (curr_kf_frame_kp_id, track)
                    break

        corr = {}
        for keyframe, matches in kf_matches.items():
            if not matches:
                continue

            num_matches = len(matches)
            kp_old = np.zeros([2, num_matches])
            kp_new = np.zeros([2, num_matches])
            tracks = [None] * num_matches

            for i, (new_kp_id, (old_kp_id, track)) in enumerate(matches.items()):
                kp_old[:, [i]] = keyframe.get_keypoint(old_kp_id).point()
                kp_new[:, [i]] = new_keyframe.get_keypoint(new_kp_id).point()
                tracks[i] = track

            corr[keyframe] = (kp_old, kp_new, tracks)

        return corr
