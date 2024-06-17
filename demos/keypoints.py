# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import io
import json

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from zipp import Path as ZipPath


def plot_keypoints_on_image(ava_dir, subject_id, base_dir, camera_id, frame_id, intrin, extrin, savefig=False, showfig=False):

    base_dir = f"{ava_dir}/{subject_id}/decoder/"
    path = ZipPath(
                base_dir + "image/" + f"cam{camera_id}.zip",
                f"cam{camera_id}/{int(frame_id):06d}.avif",
            )
    img_bytes = path.read_bytes()
    image = Image.open(io.BytesIO(img_bytes))
    
    path = ZipPath(f"{base_dir}/keypoints_3d/keypoints_3d.zip", f"{frame_id:06d}.npy")

    keypoints = np.load(io.BytesIO(path.read_bytes()))

    keypoints = keypoints.reshape(-1, 6)

    keypoints = np.append(keypoints[:, 1:4], np.ones((keypoints.shape[0], 1)), axis=1)

    twod = np.dot(np.matmul(intrin, extrin), np.transpose(keypoints))
    twod /= twod[-1]
    twod /= 4  # images have been downscaled by 4

    fig, ax = plt.subplots()
    fig.patch.set_visible(False)
    ax.axis("off")
    
    ax.set_xlim(0, 666)
    ax.set_ylim(1023, 0)

    plt.imshow(image)
    plt.scatter([twod[0]], [twod[1]], s=1, c="yellow")

    plt.box(False)

    if savefig:
        plt.savefig(f"viz/keypoints_demo-{subject_id}+{camera_id}+{frame_id}.png",bbox_inches='tight',pad_inches = 0)
    if showfig:
        plt.show()
    plt.close()

    return plt


def plot_keypoints_3d(ava_dir, subject_id, base_dir, frame_id, elev=50, azim=90, roll=0, savefig=False, showfig=False):

    ax = plt.figure().add_subplot(projection="3d")

    base_dir = f"{ava_dir}/{subject_id}/decoder/"

    path = ZipPath(f"{base_dir}/keypoints_3d/keypoints_3d.zip", f"{frame_id:06d}.npy")

    keypoints = np.load(io.BytesIO(path.read_bytes()))

    print(f"Loaded keypoints of shape {keypoints.shape}")

    keypoints = keypoints.reshape(-1, 6)

    keypoints = keypoints[:, 1:4]

    ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], s=10)
    ax.view_init(elev=elev, azim=azim, roll=roll)

    if savefig:
        plt.savefig(f"viz/keypoints3D_demo-{subject_id}+{frame_id}.png",bbox_inches='tight',pad_inches = 0)
    if showfig:
        plt.show()
    plt.close()
    
if __name__ == "__main__":
    camera_id = '401031'
    frame_id = 35665
    ava_dir = "../AVA_dataset_8TB/"
    subject_id = "20230801--1420--EJG940"
    base_dir = f"{ava_dir}/{subject_id}/decoder/"
    plot_keypoints_on_image(ava_dir, subject_id, base_dir, camera_id, frame_id, savefig=False, showfig=False)