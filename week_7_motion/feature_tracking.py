import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
from numpy import linalg
from scipy.signal import sepfir2d
from scipy import ndimage
from tqdm import tqdm

#load images
vid = []
vid.append(1.0*plt.imread("./week_7_motion/frame_1.jpg"))
vid.append(1.0*plt.imread("./week_7_motion/frame_100.jpg"))

# Downsize images to 100 pixels wide while maintaining aspect ratio


vid = [downsize_image(frame, 100) for frame in vid]

# Convert images to grayscale if they are in color
if vid[0].ndim == 3:
    vid = [np.mean(frame, axis=2) for frame in vid]


ydim, xdim = vid[0].shape
Vx = np.zeros((ydim//4+1, xdim//4+1))
Vy = np.zeros((ydim//4+1, xdim//4+1))
sz = 5 # block size: (2*sz+1) x (2*sz+1) pixels
cx = 0
for x in tqdm(range(0, xdim, 20), desc="Processing columns"):
    cy = 0
    # for y in tqdm(range(0, ydim, 4), desc="Processing rows", leave=False):
    for y in range(0, ydim, 20):
        if (x - sz >= 0 and x + sz < xdim and y - sz >= 0 and y + sz < ydim):
            blk1 = vid[0][y - sz:y + sz, x - sz:x + sz]
            mindiff = 1e10
            for u in range(x - 10, x + 10):
                for v in range(y - 10, y + 10):
                    if (u - sz >= 0 and u + sz < xdim and v - sz >= 0 and v + sz < ydim):
                        blk2 = vid[1][v - sz:v + sz, u - sz:u + sz]
                        diff = np.sum(np.abs(blk1 - blk2))
                        diffval = np.sum(diff.flatten())
                        if diff < mindiff:
                            mindiff = diff
                            Vx[cy, cx] = u - x
                            Vy[cy, cx] = v - y
        cy += 1
    cx += 1

#remove outliers
Vx = ndimage.median_filter(Vx, size=5)
Vy = ndimage.median_filter(Vy, size=5)


#display
plt.figure(figsize=(4*xdim/72, 4*ydim/72))
plt.imshow(vid[1], cmap='gray')
Ny, Nx = Vx.shape
x, Y = np.meshgrid(np.arange(0, Nx), np.arange(0, Ny))
Vy = -Vy

_ = plt.quiver(4*x, 4*Y, Vx, Vy, color='y', scale=50, alpha=0.8, width=0.005, minlength=0.1)

plt.axis('off')
plt.savefig('./week_7_motion/feature_tracking_result.png')
