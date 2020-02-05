# Image-Processing

1.	Write a function that returns value of a pixel entered 
 
%matplotlib inline
from skimage.io import imread, imsave, imshow
img = imread('https://stepik.org/media/attachments/lesson/58180/tiger-gray.png')

def get_val(x, y, z):
    v =img[x, y, z]
print(v)

x = int(input("x ="))
y = int(input("y ="))
z = int(input("z ="))

get_val(x, y, z)
 

2.	Write a function that sets value of a pixel entered 
 
%matplotlib inline
from skimage.io import imread, imsave, imshow
img = imread('https://stepik.org/media/attachments/lesson/58180/tiger-gray.png')


def get_val(x, y, z):
    v =img[x, y, z]
    print(v)
def set_val(x, y, z, val):
    v = img[x, y, z] = val
    print(v)
    
x = int(input("x ="))
y = int(input("y ="))
z = int(input("z ="))
val = int(input("val ="))

set_val(x, y, z, val)

 

Print in the end of the code is used in order to check whether the value was entered.

3.	Write a function that creates a copy of a image: 

%matplotlib inline
from skimage.io import imread, imsave, imshow
img = imread('https://stepik.org/media/attachments/lesson/58180/tiger-gray.png')
def copy_pic(img):
    img_copy = img.copy()
copy_pic(img)
imshow(img_copy)

 
 
 
4.	Write a function that turns rgb image to grayscale using relative luminance formula 
 
Formula = 0.21 R + 0.72 G + 0.07 B.

%matplotlib inline
from skimage.io import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rgb_to_gray(img):
        grayImage = np.zeros(img.shape)
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])

        R = (R *.21)
        G = (G *.72)
        B = (B *.07)

        Avg = (R+G+B)
        grayImage = img

        for i in range(3):
           grayImage[:,:,i] = Avg

        return grayImage       

image = mpimg.imread("https://stepik.org/media/attachments/lesson/58180/tiger-color.png")   
grayImage = rgb_to_gray(image)  
plt.imshow(grayImage)
plt.show()

 

5.	Write a function that shifts pixel values. It should add v value 	to every pixel in channel c	 in the image 

%matplotlib inline
from skimage.io import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def shift_pix_val(img, channel, value):
    shifted_pic = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])
    
    if channel == 0:
        R = (R + value)
    if channel == 1:
        G = (G + value)
    if channel == 2:
        B = (B + value)
    
    pic = (R+G+B)
    shifted_pic = img

    for i in range(3):
        shifted_pic[:,:,i] = pic

    return shifted_pic 

image = mpimg.imread("https://stepik.org/media/attachments/lesson/58180/tiger-color.png")
channel = int(input("channel ="))
value = int(input("value ="))
shifted_pic = shift_pix_val(image, channel, value)  
plt.imshow(shifted_pic)
plt.show()
  

As expected out of range error appears. 

6.	There should be a problem after adding some constant values to pixels. Solve it with sol function. 

%matplotlib inline
from skimage.io import imread, imshow, imsave
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def sol(img, channel, value):
    shifted_pic = np.zeros(img.shape)
    R = np.array(img[:, :, 0])
    G = np.array(img[:, :, 1])
    B = np.array(img[:, :, 2])
    
    if channel == 0:
        if R + value >= 255:
            R = 255
        elif channel == 0 and R + value <= 0:
            R = 0
        else:
            R = (R + value)
    
    if channel == 1:
        if G + value >= 255:
            G = 255
        elif G + value <= 0:
            G = 0
        else: 
            G = (G + value)
            
    if channel == 2:
        if B + value >= 255:
            B = 255
        elif B + value <= 0:
            B = 0
        else:
            B = (B + value)
        
    
    pic = (R+G+B)
    shifted_pic = img

    for i in range(3):
        shifted_pic[:,:,i] = pic

    return shifted_pic
    
image = mpimg.imread("https://stepik.org/media/attachments/lesson/58180/tiger-color.png")
channel = int(input("channel ="))
value = int(input("value ="))
shifted_pic = sol(image, channel, value)  
plt.imshow(shifted_pic)
plt.show()

Salt and Pepper Noise 

import numpy as np
import random
import cv2

def sp_noise(image,prob):
    
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

image = cv2.imread('kermitbnw.jpg',0) 
noise_img = sp_noise(image,0.05)
cv2.imwrite('sp_noise.jpg', noise_img)

 

 	 

 
 
Get Rid of Salt & Pepper Noise

import cv2

im = cv2.imread('sp_noise.jpg',0)
median_blur= cv2.medianBlur(im, 3)
cv2.imshow('sp_noise.jpg', im)  
cv2.imshow('median_blur', median_blur)  

cv2.waitKey()
cv2.destroyAllWindows()
 
 		 

Add Gaussian Noise

import numpy as np
import os
import cv2
from skimage.io import imread, imsave, imshow


def gauss_noise(image):
   
      row,col,ch= image.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = image + gauss
      return noisy

image = cv2.imread('teletubbies-joy-division-black-and-white.jpg') 
noise_img = gauss_noise(image)
cv2.imwrite('gauss_n.jpg', noise_img)


Get Rid of Gaussian Noise 

import numpy as np
import cv2 
from matplotlib import pyplot as plt
img = cv2.imread('additivegaussiannoise_large.jpg')

dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
plt.subplot(121),plt.imshow(img)
plt.subplot(122),plt.imshow(dst)
plt.show()
 
 
 

Fourier Transform, Add Periodic Noise, Remove Periodic Noise

import cv2
import numpy as  np
from matplotlib import pyplot as plt

img = cv2.imread("teletubbies-joy-division-black-and-white.jpg", cv2.IMREAD_GRAYSCALE)

f = np.fft.fft2(img)			
fshift = np.fft.fftshift(f)
magnitude_spectrum0 = 20*np.log(np.abs(fshift))
magnitude_spectrum0 = np.asarray(magnitude_spectrum0, dtype=np.uint8)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

rows, cols = img.shape
crow,ccol = rows/2 , cols/2

mask = np.zeros((rows,cols,2),np.uint8)
mask[int(crow-0):int(crow+750), int(ccol-0):int(ccol+750)] = 1  

fshift = dft_shift*mask 
f_ishift = np.fft.ifftshift(fshift)  
img_back = cv2.idft(f_ishift)		
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

mask2 = np.zeros((rows,cols,2),np.uint8)
mask2[int(crow-30):int(crow+30), int(ccol-30):int(ccol+30)] = 1 

fshift2 = dft_shift*mask2 
f_ishift2 = np.fft.ifftshift(fshift2)  
img_back2 = cv2.idft(f_ishift2)		
img_back2 = cv2.magnitude(img_back2[:,:,0],img_back2[:,:,1])

mag_spectrum = [img, img_back,magnitude_spectrum0,img_back2]
filter_name = ['input image', 'Periodic Noise', 'FFT','Remove Periodic Noise']
for i in range(4):
    plt.subplot(2,3,i+1),plt.imshow(mag_spectrum[i],cmap = 'gray')
    plt.title(filter_name[i]), plt.xticks([]), plt.yticks([])
plt.show()
 
 	 	 	 
