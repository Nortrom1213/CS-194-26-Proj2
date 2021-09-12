from scipy.signal import convolve2d
from scipy.signal import unit_impulse
from scipy.ndimage.interpolation import rotate
from skimage import img_as_uint
from skimage import img_as_float
import skimage as sk
import cv2
import skimage.io as io
import skimage.data as data
import skimage.transform as sktr
import numpy as np
from matplotlib.pyplot import hist
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import argparse



### 1.1
D_x = [[1, -1], [0, 0]]
D_y = [[1, 0], [-1, 0]]

# Helper functions to convolve the derivative on each axis
def convolve_x(src):
    return convolve2d(src, D_x)

def convolve_y(src):
    return convolve2d(src, D_y)

# Create a mask for the derivative images
def mask_image_derivative(dx, dy, threshold):
    mask_x = abs(dx[:,:]) > threshold
    mask_y = abs(dy[:,:]) > threshold
    return mask_x + mask_y

# Saves the borders produces by the derivative map
def Find_Edges(image):
    threshold = 20
    dx = convolve_x(image)
    dy = convolve_y(image)
    print()
    io.imsave("cameraman_dx.jpg", dx)
    io.imsave("cameraman_dy.jpg", dy)
    io.imsave("cameraman_GD.jpg", dx + dy)
    mask = mask_image_derivative(dx, dy, threshold)
    io.imsave("cameraman_Edge.jpg", img_as_uint(mask))

im = io.imread("cameraman.jpg")
img = im[:,:,0]
Find_Edges(img)


### 1.2

# Creates gaussian kernel
def gaussian_kernel(size, sigma):
    kernel = cv2.getGaussianKernel(size, sigma)
    return np.outer(kernel, np.transpose(kernel))

# Saves the blurred borders for the derivative map
def Find_Edges_Blur(image):
    thresh = 20
    gauss = gaussian_kernel(20, 1.5)
    blurry = convolve2d(image, gauss)
    dx = convolve_x(blurry)
    dy = convolve_y(blurry)
    io.imsave("cameraman_dx_BB.jpg", dx)
    io.imsave("cameraman_dy_BB.jpg", dy)
    io.imsave("cameraman_GD_BB.jpg", dx + dy)
    blurry_mask = mask_image_derivative(dx, dy, thresh)
    io.imsave( "cameraman_Edge_BB.jpg", img_as_uint(blurry_mask))

# Saves the blurred borders for the derivative map using a preprocessed filter (DOF)
def Find_Edges_DOG(image):
    thresh = 20
    gauss = gaussian_kernel(20, 1.5)
    kernel_dx = convolve_x(gauss)
    kernel_dy = convolve_y(gauss)
    blurry_dx =  convolve2d(image, kernel_dx)
    blurry_dy =  convolve2d(image, kernel_dy)
    io.imsave("cameraman_dx_DOG.jpg", blurry_dx)
    io.imsave("cameraman_dy_DOG.jpg", blurry_dy)
    io.imsave("cameraman_GD_DOG.jpg", blurry_dx + blurry_dy)
    dof_blurry_mask = mask_image_derivative(blurry_dx, blurry_dy, thresh)
    io.imsave("cameraman_Edge_DOG.jpg", img_as_uint(dof_blurry_mask))

Find_Edges_Blur(img)
Find_Edges_DOG(img)


### 2.1

unit = unit_impulse((20, 20), 'mid')

def normalize(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))

def sharpen(image, alpha = 3):
    dim1, dim2, dim3 = np.shape(image)
    result = np.ndarray((dim1, dim2, dim3))
    gauss = gaussian_kernel(20, 1.5)
    sharp_kernel = (1 + alpha) * unit - alpha * gauss
    for i in range(3):
        current_channel = image[:, :, i]
        sharp = convolve2d(current_channel, sharp_kernel, mode='same', boundary='symm')
        result[:, :, i] = normalize(sharp)
    #io.imsave("taj_high_frequency.jpg", result)
    io.imsave("taj_sharpen.jpg", image + result)

img = io.imread("taj.jpg")
sharpen(img, 2.5)

### 2.2

def hybrid(img1, img2):
    kernel_1 = gaussian_kernel(25, s1)
    # Blur every channel
    img1_blurry_0 = convolve2d(img1[:, :, 0], kernel_1, mode="same")
    img1_blurry_1 = convolve2d(img1[:, :, 1], kernel_1, mode="same")
    img1_blurry_2 = convolve2d(img1[:, :, 2], kernel_1, mode="same")
    blurry_img1 = np.dstack([img1_blurry_0, img1_blurry_1, img1_blurry_2])

    kernel_2 = gaussian_kernel(25, s2)
    # Blur every channel
    img2_blurry_0 = convolve2d(img2[:, :, 0], kernel_2, mode="same")
    img2_blurry_1 = convolve2d(img2[:, :, 1], kernel_2, mode="same")
    img2_blurry_2 = convolve2d(img2[:, :, 2], kernel_2, mode="same")
    blurry_img2 = np.dstack([img2_blurry_0, img2_blurry_1, img2_blurry_2])

    # Get high frequencies for image 2
    high_freq = img2 - blurry_img2

    result = blurry_img1 / 2 + high_freq / 2
    io.imsave("hybrid.jpg", result)

img1 = io.imread("DerekPicture.jpg")
img2 = io.imread("nutmeg.jpg")

img1 = img1 / np.max(im1)
img2 = img2 / np.max(im2)
img1, img2 = align_images(img1, img2)