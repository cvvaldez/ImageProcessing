import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('Dataset1/Image013.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)

ret, thresh = cv.threshold(cl1,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
# ret,thresh = cv.threshold(cl1,0,255,cv.THRESH_TOZERO)
# ret,thresh = cv.threshold(cl1,127,255,cv.THRESH_BINARY)
# ret,thresh = cv.threshold(cl1,127,255,cv.THRESH_BINARY_INV)
# ret,thresh = cv.threshold(cl1,127,255,cv.THRESH_TRUNC)
# ret,thresh = cv.threshold(cl1,127,255,cv.THRESH_TOZERO)
# ret,thresh = cv.threshold(cl1,127,255,cv.THRESH_TOZERO_INV)

cv.imshow('Otsu', thresh)
cv.waitKey(0)
cv.destroyAllWindows()

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 3)

cv.imshow('Opening', opening)
cv.waitKey(0)
cv.destroyAllWindows()

# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)

cv.imshow('Background', sure_bg)
cv.waitKey(0)
cv.destroyAllWindows()

# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.3*dist_transform.max(),255,0)

cv.imshow('foreground', sure_fg)
cv.waitKey(0)
cv.destroyAllWindows()

# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)

cv.imshow('Unknown', unknown)
cv.waitKey(0)
cv.destroyAllWindows()

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)

# Add one to all labels so that sure background is not 0, but 1
markers = markers+1

# Now, mark the region of unknown with zero
markers[unknown==255] = 0

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]

cv.imshow('Final segmentation', img)
cv.waitKey(0)
cv.destroyAllWindows()