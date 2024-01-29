import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import os
import math
import sklearn.cluster

extensions = ['jpg', 'webp', 'AVIF', 'HEIC']

qualities = [12, 25, 50, 70, 90, 100]

color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

img_size = 1334 * 2048

# Plot SSIM
fig, ax = plt.subplots()
for i, ext in enumerate(extensions):
    with open(f'selectedIMGFiles_compressed/{ext}_ssim.json') as f:
        ssims = json.load(f)
    xs = []
    ys = []
    for file in ssims:
        for quality in qualities:
            
            img_path = f'selectedIMGFiles_compressed/{ext}_100-{quality}/{file}'
            size = os.path.getsize(img_path) * 8    # Get the size in bits.
            x = size/img_size
            if x >= 1 or x <= 0:
                continue
            y = data[file][str(quality)]
            xs += [x]
            ys += [y]
            
    
    X = np.array([[x,y] for x,y in zip(xs,ys)])
    kmeans = sklearn.cluster.KMeans(n_clusters=5).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    centers = np.sort(centers, axis=0)
    plt.scatter(xs, ys, s=10, c=color_cycle[i], alpha=0.1, edgecolor='none')
    plt.scatter(centers[:,0], centers[:,1], label=e, s=10, c=color_cycle[i], edgecolor='none')
    plt.plot(centers[:,0], centers[:,1],c=color_cycle[i], alpha=0.5)
    plt.xlabel('bits per pixel (bpp)')
    plt.ylabel('SSIM')
    plt.xlim(0,1)
    plt.legend()
    plt.title('bbp vs SSIM')
    
plt.savefig('tmp.png')
plt.close()

fig, ax = plt.subplots()

# Plot PSNR
for i, ext in enumerate(extensions):
    with open(f'selectedIMGFiles_compressed/{ext}_psnr.json') as f:
        psnrs = json.load(f)
    xs = []
    ys = []
    for file in psnrs:
        for quality in qualities:
            img_path = f'selectedIMGFiles_compressed/{ext}_100-{quality}/{file}'
            size = os.path.getsize(img_path) * 8    # Get the size in bits.
            x = size/img_size
            y = data[file][str(quality)]
            if x >= 1 or x <= 0:
                continue
            xs += [x]
            ys += [y]
            
    X = np.array([[x,y] for x,y in zip(xs,ys)])
    kmeans = sklearn.cluster.KMeans(n_clusters=5).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    centers = np.sort(centers, axis=0)
    plt.scatter(xs, ys, s=10, c=color_cycle[i], alpha=0.1, edgecolor='none')
    plt.scatter(centers[:,0], centers[:,1], label=e, s=10, c=color_cycle[i], edgecolor='none')
    plt.plot(centers[:,0], centers[:,1],c=color_cycle[i], alpha=0.5)
    plt.xlabel('bits per pixel (bpp)')
    plt.ylabel('PSNR')
    plt.xlim(0,1)
    plt.legend()
    plt.title('bbp vs PSNR')
    
plt.savefig('tmp-psnr.png')
plt.close()

# Plot decompression time
fig, ax = plt.subplots()

for i, ext in enumerate(extensions):
    with open(f'selectedIMGFiles_compressed/{ext}_time-compress.json') as f:
        data = json.load(f)
    xs = []
    ys = []
    for file in data:
        for quality in qualities:
            
            img_path = f'selectedIMGFiles_compressed/{ext}_100-{quality}/{file[:-3]}{e}'
            size = os.path.getsize(img_path) * 8    # Get the size in bits.
            x = size/img_size
            y = data[file][str(quality)]
            if x >= 1 or x <= 0:
                continue
            xs += [x]
            ys += [y]
            
    X = np.array([[x,y] for x,y in zip(xs,ys)])
    kmeans = sklearn.cluster.KMeans(n_clusters=5).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    centers = np.sort(centers, axis=0)
    plt.scatter(xs, ys, s=10, c=color_cycle[i], alpha=0.1, edgecolor='none')
    plt.scatter(centers[:,0], centers[:,1], label=e, s=10, c=color_cycle[i], edgecolor='none')
    plt.plot(centers[:,0], centers[:,1],c=color_cycle[i], alpha=0.5)
    plt.xlabel('bits per pixel (bpp)')
    plt.xlim(0,1)
    plt.ylabel('time (s)')
    plt.legend()
    plt.title('bbp vs Compression Time')
    
plt.savefig('tmp-compressTime.png')
plt.close()
    

# Plot decompression time
fig, ax = plt.subplots()

for i, ext in enumerate(extensions):
    with open(f'selectedIMGFiles_compressed/{ext}_time-decompress.json') as f:
        data = json.load(f)
    xs = []
    ys = []
    for file in data:
        for quality in qualities:
            
            img_path = f'selectedIMGFiles_compressed/{ext}_100-{quality}/{file}'
            size = os.path.getsize(img_path) * 8    # Get the size in bits.
            x = size/img_size
            y = data[file][str(quality)]
            if x >= 1 or x <= 0:
                continue
            xs += [x]
            ys += [y]
            
    X = np.array([[x,y] for x,y in zip(xs,ys)])
    kmeans = sklearn.cluster.KMeans(n_clusters=5).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    centers = np.sort(centers, axis=0)
    plt.scatter(xs, ys, s=10, c=color_cycle[i], alpha=0.1, edgecolor='none')
    plt.scatter(centers[:,0], centers[:,1], label=e, s=10, c=color_cycle[i], edgecolor='none')
    plt.plot(centers[:,0], centers[:,1],c=color_cycle[i], alpha=0.5)
    plt.xlabel('bits per pixel (bpp)')
    plt.ylabel('time (s)')
    plt.xlim(0,1)
    plt.legend()
    plt.title('bbp vs Decompression Time')
    
plt.savefig('tmp-decompressTime.png')
plt.close()
    
    
