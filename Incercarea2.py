import cv2
import numpy as np
import matplotlib.pyplot as plt

ksize = 101  # size of the Gabor kernel
sigma_i = [10, 30, 45] # sigma_i = [10, 30, 45]
theta_i = [1/4.*np.pi,2/4.*np.pi,3/4*np.pi, 4/4.*np.pi]  #orientation of the Gabor function.
lamda_i = [10,60,100 ]#  width of the strips of Gabor function
gamma = 0.4  # gamma controls the height of the Gabor function
phi = 0  # Phase offset.

# pregatire bank kernels
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
plt.axis('off')
plt.imshow(img, cmap='gray')
plt.title('Imagine originala')

#Afisare banca de filtre:
plt.figure()
n = len(kernels)
for i in range(1, n):
    plt.subplot(6,6,i)
    plt.axis('off')
    plt.imshow(kernels[i], cmap='gray')
    #plt.text(-450,400,gaborParams[i], fontsize=8)

#Afisare imagini filtrate:
plt.figure()
n = len(kernels)
for i in range(1, n):
    plt.subplot(6,6,i)
    plt.axis('off')
    plt.imshow(imagini_filtrate[i], cmap='gray')

plt.show()
