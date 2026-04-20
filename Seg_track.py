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
from skimage import data
from skimage.segmentation import watershed
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import scipy.ndimage as nd
from help import mean_dataset, std_dataset
from skimage.color import label2rgb

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
def watershed_seg(nb=10, k=1):
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
        markers = np.zeros_like(img_read) #empty image
        #Assign labels to each pixel in the image with respect to their intensity
        # To choose thresholds; we compute mean and std of our train set: using 
        # mean_dataset() and std_dataset() in help.py
        # low_threshold = mean - k * std
        # high_threshold = mean + k * std
        low_th = mean_dataset(images) - k * std_dataset(images) #k = 1
        high_th = mean_dataset(images) + k * std_dataset(images) # k = 1
        markers[img_read < float(low_th)] = 1 
        markers[img_read > float(high_th)] = 2 
        #print(f"Markers: {markers}")
        #print(elevation_map)
        #plt.imshow(markers, cmap='gray')
        #plt.title('markers')
        #plt.show()
        
        # Perform watershed region segmentation 
        segmentation = watershed(elevation_map, markers)
        #segmentation = cv.normalize(segmentation, None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
        #print(segmentation.max())
        #cv.imwrite(os.path.join(save_path, f"watershed_seg/seg_{i}.png"), segmentation)
        
        #plt.imshow(segmentation)
        #plt.title('Watershed segmentation')
        #plt.show()
        
        # plot overlays and contour
        segmentation = nd.binary_fill_holes(segmentation - 1)
        label_rock, _ = nd.label(segmentation)
        # overlay image with different labels
        image_label_overlay = label2rgb(label_rock, image=img_read)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 16), sharey=True)
        ax1.imshow(img_read)
        ax1.contour(segmentation, [0.8], linewidths=1.8, colors='w')
        ax2.imshow(image_label_overlay)
        plt.show()
        #fig.subplots_adjust(**margins)
        


#canny_seg()
watershed_seg(k=1)
#print(os.path.join(save_path, f"binary-seg/seg_{1}.tif"))

# sources: https://www.geeksforgeeks.org/machine-learning/region-and-edge-based-segmentaion/
# https://www.geeksforgeeks.org/computer-vision/image-segmentation-techniques-and-applications/