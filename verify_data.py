import cv2 as cv
import glob
import os
import numpy as np
from math import log10, sqrt


def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr


path1 = "DIC-C2DH-HeLa-PNG/train/images/01"
path2 = "DIC-C2DH-HeLa-PNG/train/images/02"

images1 = sorted(glob.glob(os.path.join(path1, '*.png')))
images2 = sorted(glob.glob(os.path.join(path2, '*.png')))

nb_img = len(images1)
psnr_cum = 0
for i in range(nb_img):
    img1 = cv.imread(images1[i])
    img2 = cv.imread(images2[i])
    
    psnr_val = PSNR(img1, img2)
    psnr_cum += psnr_val
    print(f"processing {images1[i]} and {images2[i]} -- PSNR between images {i} is: {psnr_val}")

print(f"the average PSNR is : {psnr_cum / nb_img}")
