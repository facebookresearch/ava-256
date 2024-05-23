import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import os
import math
import sklearn.cluster
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i", "--input", default="selectedIMGFiles_compressed/", help="output folder to store compressed files"
)

args = parser.parse_args()


variables = [
    ("ssim", "SSIM"),
    ("psnr", "PSNR"),
    ("time-compress", "Compression Time"),
    ("time-decompress", "Decompression Time"),
]

for var in variables:
    fig, ax = plt.subplots()
    for i, ext in enumerate(extensions):
        with open(f"{args.output}{ext}_{var[0]}.json") as f:
            data = json.load(f)
        xs = []
        ys = []
        for file in data:
            for quality in qualities:

                img_path = f"{args.input}{ext}_100-{quality}/{file}"
                size = os.path.getsize(img_path) * 8  # Get the size in bits.
                x = size / img_size
                if x >= 1:
                    continue
                if x <= 0:
                    raise Exception("bpp cannot be below Zero")
                y = data[file][str(quality)]
                xs += [x]
                ys += [y]

        X = np.array([[x, y] for x, y in zip(xs, ys)])
        kmeans = sklearn.cluster.KMeans(n_clusters=5).fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        centers = np.sort(centers, axis=0)
        plt.scatter(xs, ys, s=10, c=color_cycle[i], alpha=0.1, edgecolor="none")
        plt.scatter(centers[:, 0], centers[:, 1], label=e, s=10, c=color_cycle[i], edgecolor="none")
        plt.plot(centers[:, 0], centers[:, 1], c=color_cycle[i], alpha=0.5)
        plt.xlabel("bits per pixel (bpp)")
        plt.ylabel(var[1])
        plt.xlim(0, 1)
        plt.legend()
        plt.title(f"bbp vs {var[1]}")

    plt.savefig(f"tmp-{var[0]}.png")
    plt.close()
