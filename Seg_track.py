# You can find the dataset heare: https://celltrackingchallenge.net/2d-datasets/
# Open source project:  https://github.com/z-x-yang/Segment-and-Track-Anything

import cv2 as cv
import numpy as np
import os
from PIL import Image
import glob

path = "DIC-C2DH-HeLa/01"
save_path = "output"
images = glob.glob(os.path.join(path, "*.tif"))

#visualize some samples of the dataset
nb = 10
for i in range(nb):
    #print(f"List images {i}: {images[i]}")
    img_read = cv.imread(images[i])
    print(f"Image matrix max value: {img_read.max()}")
    #binary segmentation
    _, img_seg = cv.threshold(img_read, 120, 255, cv.THRESH_BINARY)
    cv.imwrite(os.path.join(save_path, f"seg_{i}.tif"), img_seg)
    cv.imshow(f"{nb} images of the dataset", np.array(Image.open(images[i])))
    cv.waitKey(0)
cv.destroyAllWindows()

# Binary segmentation
