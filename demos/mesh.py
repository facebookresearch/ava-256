import numpy as np
import io
from PIL import Image
import pillow_avif
import matplotlib.pyplot as plt
import matplotlib.collections as mcoll
import pickle
from zipp import Path as ZipPath
from plyfile import PlyData
import plyfile
from utils import *
import mpl_toolkits

def plot_mesh_on_image(ava_dir, subject_id, base_dir, camera_id, frame_id, savefig=False, showfig=False):

    base_dir = f"{ava_dir}/{subject_id}/decoder/"
    path = ZipPath(
        base_dir + "image/" + f"cam{camera_id}.zip",
        f"cam{camera_id}/{int(frame_id):06d}.avif",
    )
    img_bytes = path.read_bytes()
    image = Image.open(io.BytesIO(img_bytes))

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

    path = ZipPath(f"{base_dir}/head_pose/head_pose.zip", f"{frame_id:06d}.npy")

    head_pose = np.load(io.BytesIO(path.read_bytes()))

    vertices = np.append(vertices, np.ones((vertices.shape[0], 1)), axis=1)
    vertices = np.matmul(head_pose, np.transpose(vertices))
    vertices = np.append(vertices, np.ones((1, vertices.shape[1])), axis=0)

    twod = np.dot(np.matmul(intrin, extrin), vertices)
    twod /= twod[-1]
    twod /= 4  # images have been downscaled by 4

    topology_path = "./assets/face_topology.obj"

    dotobj = load_obj(topology_path)

    vi = dotobj["vi"]

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis("off")

    plt.imshow(image)
    
    xs = twod[0, vi]
    ys = twod[1, vi]
    segments = np.array(list(zip(xs, ys))).swapaxes(1,2)
    print(segments.shape)
    line_segments = mcoll.LineCollection(segments, colors='blue', linewidth=0.1)

    ax.add_collection(line_segments)

    plt.box(False)

    if savefig:
        plt.savefig("viz/mesh_demo-{subject_id}+{camera_id}+{frame_id}.png")
    if showfig:
        plt.show()

    return plt


def plot_mesh_3d(ava_dir, subject_id, base_dir, frame_id, elev=50, azim=90, roll=0, savefig=False, showfig=False):

    ax = plt.figure().add_subplot(projection="3d")

    base_dir = f"{ava_dir}/{subject_id}/decoder/"

    path = ZipPath(f"{base_dir}/kinematic_tracking/registration_vertices.zip", f"{frame_id:06d}.ply")

    ply_bytes = path.read_bytes()
    ply_bytes = io.BytesIO(ply_bytes)
    plydata = PlyData.read(ply_bytes)
    vertices = plydata["vertex"].data
    vertices = np.array([list(element) for element in vertices])

    topology_path = "./assets/face_topology.obj"

    dotobj = load_obj(topology_path)

    vi = dotobj["vi"]
    
    vertices = vertices.swapaxes(0,1)
        
    xs = vertices[0, vi]
    ys = vertices[1, vi]
    zs = vertices[2, vi]
    segments = np.array(list(zip(xs, ys, zs))).swapaxes(1,2)
    line_segments = mpl_toolkits.mplot3d.art3d.Line3DCollection(segments, colors='blue', linewidth=0.1)

    ax.add_collection(line_segments)
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_zlim([-200, 200])


    ax.view_init(elev=elev, azim=azim, roll=roll)

    if savefig:
        plt.savefig("viz/mesh3D_demo-{subject_id}+{frame_id}.png")
    if showfig:
        plt.show()
