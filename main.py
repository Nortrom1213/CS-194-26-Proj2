from align_image_code import align_images
from scipy.signal import unit_impulse
from scipy.signal import convolve2d
from skimage.color import rgb2gray
import skimage.io as io
import numpy as np
import cv2


D_x = [[1, -1], [0, 0]]
D_y = [[1, 0], [-1, 0]]

def convolve_x(src):
    return convolve2d(src, D_x)

def convolve_y(src):
    return convolve2d(src, D_y)

### 1.1
def mask_noise(dx, dy, threshold):
    mask_x = abs(dx[:,:]) > threshold
    mask_y = abs(dy[:,:]) > threshold
    return mask_x + mask_y

def Find_Edges(image, outputname, threshold):
    dx = convolve_x(image)
    dy = convolve_y(image)
    io.imsave(outputname + "_dx.jpg", dx)
    io.imsave(outputname + "_dy.jpg", dy)
    io.imsave(outputname + "_GD.jpg", dx + dy)

    result = mask_noise(dx, dy, threshold)
    io.imsave(outputname + "_Edge.jpg", result)

im = io.imread("cameraman.png")
img = im[:,:,0]
Find_Edges(img, "cameraman", 20)

### 1.2
def gaussian_kernel(size, sigma):
    kernel = cv2.getGaussianKernel(size, sigma)
    return np.outer(kernel, np.transpose(kernel))

def Find_Edges_Blur(image, outputname, threshold):
    gauss = gaussian_kernel(20, 1.5)
    blur = convolve2d(image, gauss)
    dx = convolve_x(blur)
    dy = convolve_y(blur)

    io.imsave(outputname + "_dx_BB.jpg", dx)
    io.imsave(outputname + "_dy_BB.jpg", dy)
    io.imsave(outputname + "_GD_BB.jpg", dx + dy)

    result = mask_noise(dx, dy, threshold)
    io.imsave(outputname + "_Edge_BB.jpg", result)

def Find_Edges_DOG(image, outputname, threshold):
    gauss = gaussian_kernel(20, 1.5)
    dx = convolve_x(gauss)
    dy = convolve_y(gauss)
    blur_dx = convolve2d(image, dx)
    blur_dy = convolve2d(image, dy)

    io.imsave(outputname + "_dx_DOG.jpg", blur_dx)
    io.imsave(outputname + "_dy_DOG.jpg", blur_dy)
    io.imsave(outputname + "_GD_DOG.jpg", blur_dx + blur_dy)

    result = mask_noise(blur_dx, blur_dy, threshold)
    io.imsave(outputname + "_Edge_DOG.jpg", result)

Find_Edges_Blur(img, "cameraman", 20)
Find_Edges_DOG(img, "cameraman", 20)

### 2.1
unit = unit_impulse((20, 20), 'mid')
def Blur(img, size, sigma):
    kernel = gaussian_kernel(size, sigma)
    img_blur_0 = convolve2d(img[:, :, 0], kernel, mode="same")
    img_blur_1 = convolve2d(img[:, :, 1], kernel, mode="same")
    img_blur_2 = convolve2d(img[:, :, 2], kernel, mode="same")
    return np.dstack([img_blur_0, img_blur_1, img_blur_2])

def normalize(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img))

def sharpen(image, alpha, outputname):
    dim1, dim2, dim3 = np.shape(image)
    result = np.ndarray((dim1, dim2, dim3))
    gauss = gaussian_kernel(20, 1.5)
    sharp_kernel = (1 + alpha) * unit - alpha * gauss
    for i in range(3):
        current_channel = image[:, :, i]
        sharp = convolve2d(current_channel, sharp_kernel, mode='same', boundary='symm')
        result[:, :, i] = normalize(sharp)
    io.imsave(outputname, image + result)

img = io.imread("taj.jpg")
sharpen(img, 3, "taj_sharpen.jpg")

img = io.imread("pentakill.jpg")
blur_img = Blur(img, 20, 1.5)
io.imsave("pentakill_blur.jpg", blur_img)
sharpen(blur_img, 3, "pentakill_sharpen.jpg")

### 2.2
def hybrid(img1, img2):
    img1, img2 = align_images(img1, img2)
    img1 = img1 / np.max(img1)
    img2 = img2 / np.max(img2)
    blur_img1 = Blur(img1, 20, 1.5)
    blur_img2 = Blur(img2, 20, 1.5)

    high_freq = img2 - blur_img2
    result = blur_img1 / 2 + high_freq / 2
    io.imsave("hybrid.jpg", result)

img1 = io.imread("Biden.jpg")
img2 = io.imread("Trump.jpg")

#hybrid(img1, img2)


### 2.3
def gaussian_stack(image, size, sigma):
    stack = [image]
    for i in range(0,5):
        blur = Blur(stack[i], size, sigma)
        stack.append(blur)
    return stack[1:]

def laplacian_stack(image, size, sigma):
    stack = [image]
    gauss_stack = gaussian_stack(image, size, sigma)
    cur = image
    for i in range(0,4):
        sharp = (cur - gauss_stack[i])
        stack.append(sharp / 255)
        cur = gauss_stack[i]

    stack.append(gauss_stack[-1]/255)
    return stack[1:]

def show_stack(image, outputname):
    gauss_stack = gaussian_stack(image, 20, 5)
    gauss_stack = [gauss for gauss in gauss_stack]
    lap_stack = laplacian_stack(image, 20, 5)
    lap_stack = [lap for lap in lap_stack]

    gauss_output = np.concatenate(gauss_stack[:], axis=1)
    io.imsave(outputname + "_gaussianStack.jpg", gauss_output)
    lapl_output = np.concatenate(lap_stack[:], axis=1)
    io.imsave(outputname + "_laplacianStack.jpg", lapl_output)

img = io.imread("Thanos+CAT.jpg")
show_stack(img, "TC")

### 2.4
def blend(img1, img2, mask):
    mask = rgb2gray(mask)
    mask = mask[:,:] > 0.2
    mask = np.dstack([mask, mask, mask])
    lapl_img1 = laplacian_stack(img1, 30, 30)
    lapl_img2 = laplacian_stack(img2, 30, 30)
    gauss_mask = gaussian_stack(mask, 20, 10)

    output = []
    for i in range(0, len(lapl_img1)):
        output.append(gauss_mask[i] * lapl_img1[i] + (1 - gauss_mask[i]) * lapl_img2[i])

    output = sum(output)

    return output

img1 = io.imread("water.jpg")
img2 = io.imread("akali.jpg")
mask = io.imread("akali_mask.jpg")
result = blend(img1, img2, mask)
io.imsave("Blend4.jpg", result)