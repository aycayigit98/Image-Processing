import numpy as np
import os
import cv2
from skimage.io import imread, imsave, imshow

img = imread('face.png')

def gauss_noise(img):
   
      row,col,ch= img.shape
      mean = 0
      var = 0.1
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,ch))
      gauss = gauss.reshape(row,col,ch)
      noisy = img + gauss
      return noisy

gauss_noise(img)
imshow(gauss_noise(img))
