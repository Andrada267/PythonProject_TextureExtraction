import cv2
import numpy as np
import matplotlib.pyplot as plt
ksize = 50  # Use size that makes sense to the image and fetaure size. Large may not be good.
# On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
sigma_i = [10, 30, 45] # Large sigma on small features will fully miss the features.
theta_i = [1/4.*np.pi,2/4.*np.pi,3/4*np.pi, 4/4.*np.pi]  # /4 shows horizontal 3/4 shows other horizontal.
lamda_i = [30,60,100 ]# 1/4 works best for angled.
gamma = 0.4  # Value of 1 defines spherical. Calue close to 0 has high aspect ratio
# Value of 1, spherical may not be ideal as it picks up features from other regions.
phi = 0  # Phase offset.

kernels = []
gaborParams = []

for sigma in sigma_i:
   for lamda in lamda_i:
      for theta in theta_i:
          gaborParam = 'theta=%.2f\n sigma=%d \nlamda=%d' % (theta, sigma, lamda)
          kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
          kernels.append(kernel)
          gaborParams.append(gaborParam)

img = cv2.imread('images/caramizi.jpg')  # import image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

img = cv2.imread('images/sandstone.tif')#import image
kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)#imagine filtrata
kernel_resized = cv2.resize(kernel, (500, 500))  # Resize image

fig = plt.figure(figsize=(12, 12))

ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap = 'gray')
ax1.title.set_text('Original Image')

ax2 = fig.add_subplot(2,2,3)
ax2.imshow(kernel_resized, cmap = 'gray')
ax2.title.set_text('Kernal Resized')

ax3= fig.add_subplot(2,2,2)
ax3.imshow(kernel, cmap = 'gray')
ax3.title.set_text('Kernal')

ax4 = fig.add_subplot(2,2,4)
ax4.imshow(fimg, cmap = 'gray')
ax4.title.set_text('Filtred Image')

plt.show()
