import open3d as o3d
from dataset import read_dataset


def visualize_map(map, axis_size=1):
    poses = map.get_keyframe_poses()
    p, c = map.get_pointcloud()

    axes = []
    for pose in poses:
        axes.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size).transform(pose.to_matrix()))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p.T)
    pcd.colors = o3d.utility.Vector3dVector(c.T / 255)

    o3d.visualization.draw_geometries([pcd] + axes)


if __name__ == '__main__':
    _, sfm_map = read_dataset()
    visualize_map(sfm_map, 25)
