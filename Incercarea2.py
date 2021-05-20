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

imagini_filtrate = []
for kernel in kernels:
    fimg= cv2.filter2D(img, cv2.CV_8UC3, kernel)  # imagine filtrata
    imagini_filtrate.append(fimg)

#Afisare imagine originala:
plt.figure()
plt.imshow(img, cmap='gray')
plt.title('Imagine originala')

#Afisare banca de filtre:
plt.figure()
n = len(kernels)
for i in range(1, n):
    plt.subplot(6,6,i)
    plt.axis('off')
    plt.imshow(kernels[i])
    plt.text(-45,40,gaborParams[i], fontsize=8)

#Afisare imagini filtrate:
plt.figure()
n = len(kernels)
for i in range(1, n):
    plt.subplot(6,6,i)
    plt.axis('off')
    plt.imshow(imagini_filtrate[i])

plt.show()
