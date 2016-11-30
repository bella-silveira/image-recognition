import cv2
import numpy as np
from matplotlib import pyplot as plt


# displays an image using the library pyopencv2
def show_img_cv(window_title, img):
	cv2.imshow(window_title, img)
	cv2.waitKey(0)


# displays an image using the library matplotlib
def show_img_matplot(img):
	# reminder: cv2 uses BGR instead of RGB
	plt.imshow(img)
	plt.show()

if __name__ == "__main__":

	# loading the image as a nparray 
	img_file_path = "./img/Lenna.png"
	img = cv2.imread(img_file_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # converting to RGB

	# 1)
	show_img_cv("first image", img)

	# 2)
	show_img_matplot(img)

	
