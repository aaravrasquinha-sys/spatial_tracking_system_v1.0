import json
import numpy as np
import open3d as o3d

with open('calibration.cam0.json', 'r') as f:
    data = json.load(f)
T_room_cam = np.array(data['T_room_cam'])

cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
cam_frame.transform(T_room_cam)

o3d.io.write_triangle_mesh('camera_frame.ply', cam_frame)
print("Successfully exported camera_frame.ply!")
