import demos.mesh
import io
import pickle

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import mpl_toolkits
import numpy as np
from PIL import Image
from plyfile import PlyData
from zipp import Path as ZipPath

from utils import *

camera_id = '401031'
frame_id = 31744
ava_dir = "../AVA_dataset/"
subject_id = "20230308--1352--BDF920"

base_dir = f"{ava_dir}/{subject_id}/decoder/"
# path = ZipPath(
#     base_dir + "image/" + f"cam{camera_id}.zip",
#     f"cam{camera_id}/{int(frame_id):06d}.avif",
# )
# img_bytes = path.read_bytes()
# image = Image.open(io.BytesIO(img_bytes))

path = f"{base_dir}/camera_calibration.pkl"

with open(path, "rb") as f:
    camera_calibration = pickle.load(f)

print(f"Loaded camera calibration")

params = camera_calibration[camera_id]

intrin = params["intrin"]
extrin = params["extrin"]

path = ZipPath(f"{base_dir}/kinematic_tracking/registration_vertices.zip", f"{frame_id:06d}.ply")

ply_bytes = path.read_bytes()
ply_bytes = io.BytesIO(ply_bytes)
plydata = PlyData.read(ply_bytes)
vertices = plydata["vertex"].data
vertices = np.array([list(element) for element in vertices])
print(vertices.shape)

topology_path = "./assets/face_topology.obj"

dotobj = load_obj(topology_path)

vi = dotobj["vi"]

vertices = vertices.swapaxes(0, 1)

dotobj['v'] = vertices

with open('new_mesh.obj', 'w') as f:
