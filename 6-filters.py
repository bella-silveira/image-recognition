import cv2
from matplotlib import pyplot as plt
from rootsift import RootSIFT

#img_file_path = "./img/Lenna.png"

img_file_path = "./img/cifar-10/test/truck/delivery_truck_s_000030.png"
#img_file_path2 = "./img/cifar-10/test/dog/chihuahua_s_001095.png"

#open image
img = cv2.imread(img_file_path)

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # para converter para GRAY <<<<<<<<<<<<<<<<<<
#img = cv2.resize(img,None,fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
#print img.shape

#(width, height) = img.shape

#crop_img = img[0:width/2, 0:height/2]

#fig, subs = plt.subplots(1,2)
#plt.gray()
#subs[0].imshow(img)
#subs[1].imshow(crop_img)
#plt.show()

#img2 = cv2.imread(img_file_path2)

#img = cv2.GaussianBlur(img,(5,5), 1)


#img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) # para converter para GRAY <<<<<<<<<<<<<<<<<<

## 1) blur (uniform)
#img_blur = cv2.blur(img, (3,3))

#fig, subs = plt.subplots(1,2)
# plt.gray()
#subs[0].imshow(img)
#subs[1].imshow(img_blur) 
#plt.show()


## 2) gaussian blur (normalized)
#img_gauss = cv2.GaussianBlur(img,(11,11), 1)
# fig, subs = plt.subplots(1,2)
# subs[0].imshow(img)
# subs[1].imshow(img_gauss)
# plt.show()

## 3) sobel filter
##x: [[-1,0,1],[-2,0,2],[-1,0,1]]
##y: [[1,2,1],[0,0,0],[-1,-2,-1]]
#sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,1,0,cv2.BORDER_DEFAULT)
#sobely = cv2.Sobel(img,cv2.CV_64F,0,1,1,0,cv2.BORDER_DEFAULT)
#abs_sobel_x = cv2.convertScaleAbs(sobelx)
#abs_sobel_y = cv2.convertScaleAbs(sobely)
#sobel = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
#fig, subs = plt.subplots(1,4)
#plt.gray()
#subs[0].imshow(img)
#subs[1].imshow(sobelx)
#subs[2].imshow(sobely)
#subs[3].imshow(sobel)

#plt.subplot(2,2,1),plt.imshow(img,cmap = 'gray')
#plt.title('Original'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,2),plt.imshow(sobelx,cmap = 'gray')
#plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,3),plt.imshow(sobely,cmap = 'gray')
#plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#plt.subplot(2,2,4),plt.imshow(sobel,cmap = 'gray')
#plt.title('Sobel'), plt.xticks([]), plt.yticks([])

#plt.show()


## 4) laplacian filter
## kernel1 [[0, 1, 0],[1, -4, 1],[0, 1, 0]]
## kernel2 [[0, 1, 0],[1, -8, 1],[0, 1, 0]]
#img_laplacian = cv2.Laplacian(img,cv2.CV_64F)
# fig, subs = plt.subplots(1,2)
# plt.gray()
# subs[0].imshow(img)
# subs[1].imshow(img_laplacian)
# plt.show()



## 5) diff of blur
#dob = img - img_blur
# fig, subs = plt.subplots(1,2)
# plt.gray()
# subs[0].imshow(img)
# subs[1].imshow(dob)
# plt.show()


## 6) DoG
#img_gauss1 = cv2.GaussianBlur(img,(5,5), 1)
#img_gauss2 = cv2.GaussianBlur(img_gauss1,(5,5), 1)

#dog = img - img_gauss1
#fig, subs = plt.subplots(1,2)
#plt.gray()
#subs[0].imshow(img)
#subs[1].imshow(dog)
#plt.show()


hog = cv2.HOGDescriptor("hog.xml")
h = hog.compute(img)

print h

# extract normal SIFT descriptors
#sift = cv2.xfeatures2d.SIFT_create()

#(kps, descs) = sift.detectAndCompute(img, None)
#print "SIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape)
#print descs
 

#(kps2, descs2) = sift.detectAndCompute(img2, None)
#print "SIFT2: kps=%d, descriptors=%s " % (len(kps2), descs2.shape)
 


# extract RootSIFT descriptors
#rs = RootSIFT()
#(kps, descs) = rs.compute(img, kps)
#print "RootSIFT: kps=%d, descriptors=%s " % (len(kps), descs.shape)

#kps.sort(key=lambda x:x.size ,reverse=True)

#kps2.sort(key=lambda x:x.size ,reverse=True)

#kps = kps[:2]
#kps2= kps2[:2]

#img=cv2.drawKeypoints(img,kps,img)
#img2=cv2.drawKeypoints(img2,kps2,img2)


#fig, subs = plt.subplots(1,2)
#plt.gray()
#subs[0].imshow(img)
#subs[1].imshow(img2)
#plt.show()
