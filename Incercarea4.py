import cv2
import numpy as np
import matplotlib.pyplot as plt
ksize = 5  # Use size that makes sense to the image and fetaure size. Large may not be good.
# On the synthetic image it is clear how ksize affects imgae (try 5 and 50)
lamda = 1 * np.pi / 4  # 1/4 works best for angled.
gamma = 0.4  # Value of 1 defines spherical. Calue close to 0 has high aspect ratio
# Value of 1, spherical may not be ideal as it picks up features from other regions.
phi = 0  # Phase offset.

# prepare filter bank kernels
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, phi, ktype=cv2.CV_32F)
            kernels.append(kernel)


img = cv2.imread('images/sandstone.tif')#import image
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for filtru in kernels:
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)  # imagine filtrata

fig = plt.figure(figsize=(12, 12))

ax1 = fig.add_subplot(2,2,1)
ax1.imshow(img, cmap = 'gray')
ax1.title.set_text('Original Image')

ax3= fig.add_subplot(2,2,2)
ax3.imshow(kernel, cmap = 'gray')
ax3.title.set_text('Kernal')

ax4 = fig.add_subplot(2,2,4)
ax4.imshow(fimg, cmap = 'gray')
ax4.title.set_text('Filtred Image')

plt.show()
