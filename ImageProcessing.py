import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt

def Histogram():
	images = [cv2.imread(file) for file in glob.glob("Dataset1/*.jpg")]
	d = 1
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

	for f1 in images:
		grayimg = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
		cl1 = clahe.apply(grayimg)
		filename = "Clahe/clahe_0%d.jpg" %d
		cv2.imwrite(filename, grayimg)
		d+=1
		cv2.waitKey(0)
		cv2.destroyAllWindows()

# def Fourier():
	# img = [cv2.imread(file) for file in glob.glob("Clahe/*.jpg")]
	# d = 1
	
	# for f1 in img:
		# f = np.fft.fft2(f1)
		# fshift = np.fft.fftshift(f)
		# rows, cols = f1.shape
		# f = np.fft.fft2(f1)
		# fshift = np.fft.fftshift(f)
		# rows, cols = f1.shape
		# crow,ccol = int(rows/2) , int(cols/2)
		# fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
		# f_ishift = np.fft.ifftshift(fshift)
		# img_back = np.fft.ifft2(f_ishift)
		# img_back = np.abs(img_back)
		# filename = "Fourier/fourier_0%d.jpg" %d
		# cv2.imwrite(filename, img_back)
		# d+=1
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()

def Erosion():

	images = [cv2.imread(file) for file in glob.glob("Clahe/*.jpg")]
	d = 1
	kernel = np.ones((5,5),np.uint8)
	
	for f1 in images:
		erosion = cv2.erode(f1,kernel,iterations = 1)
		filename = "Erosion/erosion_0%d.jpg" %d
		cv2.imwrite(filename, erosion)
		d+=1
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def Dilation():

	images = [cv2.imread(file) for file in glob.glob("Erosion/*.jpg")]
	d = 1
	kernel = np.ones((5,5),np.uint8)
	
	for f1 in images:
		dilation = cv2.dilate(f1,kernel,iterations = 1)
		filename = "Dilation/dilation_0%d.jpg" %d
		cv2.imwrite(filename, dilation)
		d+=1
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
def Opening():

	images = [cv2.imread(file) for file in glob.glob("Dilation/*.jpg")]
	d = 1
	kernel = np.ones((5,5),np.uint8)
	
	for f1 in images:
		opening = cv2.morphologyEx(f1, cv2.MORPH_OPEN, kernel)
		filename = "Opening/opening_0%d.jpg" %d
		cv2.imwrite(filename, opening)
		d+=1
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def Closing():

	images = [cv2.imread(file) for file in glob.glob("Opening/*.jpg")]
	d = 1
	kernel = np.ones((5,5),np.uint8)
	
	for f1 in images:
		closing = cv2.morphologyEx(f1, cv2.MORPH_CLOSE, kernel)
		filename = "Closing/closing_0%d.jpg" %d
		cv2.imwrite(filename, closing)
		d+=1
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def Gradient():

	images = [cv2.imread(file) for file in glob.glob("Clahe/*.jpg")]
	d = 1
	kernel = np.ones((5,5),np.uint8)
	
	for f1 in images:
		gradient = cv2.morphologyEx(f1, cv2.MORPH_GRADIENT, kernel)
		filename = "Gradient/Gradient_0%d.jpg" %d
		cv2.imwrite(filename, gradient)
		d+=1
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		
def LaPlacian():

	images = [cv2.imread(file) for file in glob.glob("Clahe/*.jpg")]
	d = 1
	
	for f1 in images:
		img=f1[52:308,52:308]
		print (img.shape)
		filename = "LaPlacian/laplacian_0%d.tif" %d
		cv2.imwrite(filename, img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

def Tozero():

	images = [cv2.imread(file) for file in glob.glob("Dataset1/*.jpg")]
	d = 1
	
	for f1 in images:
		ret,thresh4 = cv2.threshold(f1,127,255,cv2.THRESH_TOZERO)
		filename = "Tozero/tozero_0%d.jpg" %d
		cv2.imwrite(filename, thresh4)
		cv2.waitKey(0)
		cv2.destroyAllWindows()


Tozero()