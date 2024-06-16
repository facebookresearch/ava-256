import demos.mesh
import demos.keypoints
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
frame_id = 110585
ava_dir = "../AVA_dataset/"
subject_id = "20230309--0820--CPA784"
base_dir = f"{ava_dir}/{subject_id}/decoder/"

path = ZipPath(f"{base_dir}/kinematic_tracking/registration_vertices.zip", f"{frame_id:06d}.ply")

ply_bytes = path.read_bytes()
ply_bytes = io.BytesIO(ply_bytes)
plydata = PlyData.read(ply_bytes)
vertices = plydata["vertex"].data
vertices = np.array([list(element) for element in vertices])

topology_path = "./assets/face_topology.obj"

dotobj = load_obj(topology_path, False)

vi = dotobj["vi"]

# vertices = vertices.swapaxes(0, 1)

dotobj['v'] = vertices

with open('new_mesh.obj', 'w') as f:
    for v in dotobj['v']:
        f.write(f'v {v[0]} {v[1]} {v[2]}\n')
    for vt in dotobj['vt']:
        f.write(f'vt {vt[0]} {vt[1]}\n')



