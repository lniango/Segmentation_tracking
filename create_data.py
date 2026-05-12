import cv2 as cv
import os
import glob

images = sorted(glob.glob("DIC-C2DH-HeLa-PNG/train/images/labels_02/*.png"))

#print(images[-1][-6:-4])
cnt = 84
for image in images:
    new = f"DIC-C2DH-HeLa-PNG/train/images/labels_02/mask0{cnt}.png"
    os.rename(image, new)
    cnt += 1