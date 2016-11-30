import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# returns an image with only one of the channels
def get_img_channel(img, channel):

	img_copy = np.copy(img) 

	if channel == "r":	
		img_copy[:,:,1] = 0
		img_copy[:,:,2] = 0

	elif channel == "g":
		img_copy[:,:,0] = 0
		img_copy[:,:,2] = 0

	elif channel == "b":
		img_copy[:,:,0] = 0
		img_copy[:,:,1] = 0

	return img_copy


# generate histogram as a np.array
def hist(img):	

	R = get_img_channel(img, "r").flatten()
	G = get_img_channel(img, "g").flatten()
	B = get_img_channel(img, "b").flatten()
	
	n = 64
	hist_R,_ = np.histogram(R, bins=n)
	hist_G,_ = np.histogram(G, bins=n)	
	hist_B,_ = np.histogram(B, bins=n)

	return hist_R, hist_G, hist_B

# plot images
def draw_hist(img):
	
	fig, subs = plt.subplots(1,2)
	subs[0][0].imshow(img)
	subs[0][1].axis('off')

	R = get_img_channel(img,'r')
	G = get_img_channel(img,'g')
	B = get_img_channel(img,'b')

	subs[1][0].imshow(R)
	subs[1][1].hist(img[:,:,0].flatten(),np.arange(0,256))
	subs[1][1].set_xlim([0,256])


	subs[2][0].imshow(G)
	subs[2][1].hist(img[:,:,1].flatten(),np.arange(0,256))
	subs[2][1].set_xlim([0,256])


	subs[3][0].imshow(B)
	subs[3][1].hist(img[:,:,2].flatten(),np.arange(0,256))
	subs[3][1].set_xlim([0,256])


	plt.show()

# standardize a list
def standardize(data):
	#data[0] = 0
	mean = np.mean(data)
	std = np.std(data)
	return (data - mean)/std

# create a feature vector concatenating each image
def generate_vector(img_path):
	img = cv2.imread(img_path)	
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	(width, height,channel) = img.shape
	feature_vec = []
	for i in range(2):
		for j in range(2):
			x0 = width*i/2
			y0 = height*j/2
			crop_img = img[x0:x0+width/2, y0:y0+height/2]
			
			hist_R, hist_G, hist_B = hist(crop_img)
	
			feature_vec_block = np.hstack( [standardize(hist_R), standardize(hist_G), standardize(hist_B)])
			feature_vec = np.append(feature_vec, feature_vec_block)

	return feature_vec

def generate_vector2(img_path):
	img = cv2.imread(img_path)
	#img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # converting to GRAY <<<<<<<<<<<<<<<<<<
	hog = cv2.HOGDescriptor("hog.xml")
	h = hog.compute(img)
	return np.hstack(h)


if __name__ == "__main__":

	img_file_path = "./img/Lenna.png"
	img = cv2.imread(img_file_path)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # coverting to RGB

	# 1)
	#draw_hist(img)

	# 2) 
	#print a histogram as a vector
	#hist_R, hist_G, hist_B = hist(img)
	#print hist_R 
	#print hist_G 
	#print hist_B 

	# 3)
	#standardizing the vectors
	#hist_R, hist_G, hist_B = hist(img)
	#print standardize(hist_R)

	# 4) creating a feature vector
	#hist_R, hist_G, hist_B = hist(img)
	#feature_vec = np.hstack( [standardize(hist_R), standardize(hist_G), standardize(hist_B)])
	#print feature_vec


	# 5) processing image dataset to generate a feature vec
	test_folder = "./img/cifar-10/test"
	class_names = os.listdir(test_folder) # there are a folde for each class
		
	# processing train folder
	print "PROCESSING TEST FOLDER: "
	X = []
	y = []
	count  = 0
	for name in class_names:
		files = os.listdir(test_folder+"/"+name)
		
		# transform each file into a feature vector
		for file_name in files:
			#vec = generate_vector(test_folder+"/"+name+"/"+file_name)
			vec = generate_vector2(test_folder+"/"+name+"/"+file_name)
			#print vec.shape
			X.append(vec.tolist())

			y_vec = [0] * len(class_names) # <<<<<<<<<<<<<< HOT ENCODING REPRESENTATION <<<<<
			y_vec[class_names.index(name)] = 1
			y.append(y_vec)

			count += 1

			if count % 1000 == 0:
				print count, " images processed"


	# randomizing positions 
	np.random.seed(42)
	np.random.shuffle(X)
	np.random.seed(42)
	np.random.shuffle(y)


	# spliting the dataset in thee groups
	X_train = X[:8000]
	y_train = y[:8000]

	X_validation = X_test = X[8000: 9000]
	y_validation = y_test = y[8000: 9000]

	X_test = X[9000: ]
	y_test = y[9000: ]

	out_file = open("hist_feature_test.py","w")
	out_file.write('X_train='+str(X_train)+"\n")
	out_file.write('y_train='+str(y_train)+"\n")

	out_file.write('X_validation='+str(X_validation)+"\n")
	out_file.write('y_validation='+str(y_validation)+"\n")

	out_file.write('X_test='+str(X_test)+"\n")
	out_file.write('y_test='+str(y_test)+"\n")

	out_file.close()
