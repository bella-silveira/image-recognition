import cv2
import numpy as np
from matplotlib import pyplot as plt


# exibe uma imagem utilizando a biblioteca pyopencv2
def show_img_cv(window_title, img):
	cv2.imshow(window_title, img)
	cv2.waitKey(0)


# exibe uma imagem utilizando a biblioteca matplotlib
def show_img_matplot(img):
	# pegadinha aqui (o cv2 usa BGR ao inves de RGB)
	plt.imshow(img)
	plt.show()


if __name__ == "__main__":


	# carregando a imagem como um nparray 
	img_file_path = "./img/Lenna.png"
	img = cv2.imread(img_file_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # para converter para RGB

	# 1)
	show_img_cv("first image", img)

	# 2)
	show_img_matplot(img)

	
