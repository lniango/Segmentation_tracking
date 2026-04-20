# You can find the dataset heare: https://celltrackingchallenge.net/2d-datasets/
# Open source project:  https://github.com/z-x-yang/Segment-and-Track-Anything
# SAM2 segmentation model: https://github.com/hcmr-lab/Seg2Track-SAM2

import cv2 as cv
import numpy as np
import os
from PIL import Image
import glob
from skimage.feature import canny
from skimage.filters import sobel
from skimage import data,morphology
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import scipy.ndimage as nd

path = "DIC-C2DH-HeLa/01"
save_path = "output"
images = glob.glob(os.path.join(path, "*.tif"))

def binary_seg(nb=10):
    #visualize some samples of the dataset
    #nb = 10
    for i in range(nb):
        #print(f"List images {i}: {images[i]}")
        img_read = cv.imread(images[i])
        #print(f"Image matrix max value: {img_read.max()}")
        #binary segmentation
        _, img_seg = cv.threshold(img_read, 120, 255, cv.THRESH_BINARY)
        cv.imwrite(os.path.join(save_path, f"binary-seg/seg_{i}.png"), img_seg)
        cv.imshow(f"{nb} images of the dataset", np.array(Image.open(images[i])))
        cv.waitKey(0)
    cv.destroyAllWindows()


#Region-based segmentation
def canny_seg(nb=10):
    '''Segmentation using canny filters'''
    # apply edge segmentation
    # plot canny edge detection
    for i in range(nb):
        img_read = cv.imread(images[i])
        img_read = np.array(cv.cvtColor(img_read, cv.COLOR_BGR2GRAY)) #Use gray image
        #print(f"SHAPE of IMAGE: {img_read.shape}")
        edges = canny(img_read)
        # fill regions to perform edge segmentation
        fill_im = nd.binary_fill_holes(edges) #bool
        #print(f"DATA Type: {fill_im.dtype}")
        #Plot
        #plt.imshow(fill_im, cmap='gray')
        #plt.title('Region Filling - Canny')
        
        fill_im_uint8 = (fill_im * 255).astype(np.uint8)
        cv.imwrite(os.path.join(save_path, f"canny_seg/seg_{i}.png"), np.array(fill_im_uint8)) # write segmented image
        #plt.imshow(edges, interpolation='gaussian')
        #plt.title('Canny detector')
        #plt.show()

# Region segmentation
def region_seg(nb=10):
    for i in range(nb):
        img_read = cv.imread(images[i])
        img_read = np.array(cv.cvtColor(img_read, cv.COLOR_BGR2GRAY)) #Use gray image
        
        # Region Segmentation
        # First we print the elevation map
        elevation_map = sobel(img_read)
        #plt.imshow(elevation_map, cmap='gray')
        #plt.title('elevation map')
        #plt.show()

        # Since, the contrast difference is not much. Anyways we will perform it
        markers = np.zeros_like(img_read)
        #markers[img_read < float(50 / 255)] = 1 # 50/255
        markers[img_read > float(120 / 255)] = 2 # 120/255
        print(elevation_map)
        #plt.imshow(markers, cmap='gray')
        #plt.title('markers')
        #plt.show()

#canny_seg()
region_seg()
#print(os.path.join(save_path, f"binary-seg/seg_{1}.tif"))