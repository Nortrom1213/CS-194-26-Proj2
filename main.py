from align_image_code import align_images
from scipy.signal import unit_impulse
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from skimage import img_as_uint
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

def Find_Edges(image, output):
    threshold = 20
    dx = convolve_x(image)
    dy = convolve_y(image)
    io.imsave(output + "_dx.jpg", dx)
    io.imsave(output + "_dy.jpg", dy)
    io.imsave(output + "_GD.jpg", dx + dy)
    mask = mask_noise(dx, dy, threshold)
    io.imsave(output + "_Edge.jpg", img_as_uint(mask))

im = io.imread("cameraman.png")
img = im[:,:,0]
Find_Edges(img, "cameraman")

### 1.2
def gaussian_kernel(size, sigma):
    kernel = cv2.getGaussianKernel(size, sigma)
    return np.outer(kernel, np.transpose(kernel))

def Find_Edges_Blur(image, output):
    threshold = 20
    gauss = gaussian_kernel(20, 1.5)
    blurry = convolve2d(image, gauss)
    dx = convolve_x(blurry)
    dy = convolve_y(blurry)

    io.imsave(output + "_dx_BB.jpg", dx)
    io.imsave(output + "_dy_BB.jpg", dy)
    io.imsave(output + "_GD_BB.jpg", dx + dy)

    blurry_mask = mask_noise(dx, dy, threshold)
    io.imsave(output + "_Edge_BB.jpg", img_as_uint(blurry_mask))

def Find_Edges_DOG(image, output):
    threshold = 20
    gauss = gaussian_kernel(20, 1.5)
    kernel_dx = convolve_x(gauss)
    kernel_dy = convolve_y(gauss)
    blurry_dx = convolve2d(image, kernel_dx)
    blurry_dy = convolve2d(image, kernel_dy)

    io.imsave(output + "_dx_DOG.jpg", blurry_dx)
    io.imsave(output + "_dy_DOG.jpg", blurry_dy)
    io.imsave(output + "_GD_DOG.jpg", blurry_dx + blurry_dy)

    dof_blurry_mask = mask_noise(blurry_dx, blurry_dy, threshold)
    io.imsave(output + "_Edge_DOG.jpg", img_as_uint(dof_blurry_mask))

Find_Edges_Blur(img, "cameraman")
Find_Edges_DOG(img, "cameraman")

### 2.1
unit = unit_impulse((20, 20), 'mid')
def Blur(img, size, sigma):
    kernel = gaussian_kernel(size, sigma)
    img_blurry_0 = convolve2d(img[:, :, 0], kernel, mode="same")
    img_blurry_1 = convolve2d(img[:, :, 1], kernel, mode="same")
    img_blurry_2 = convolve2d(img[:, :, 2], kernel, mode="same")
    return np.dstack([img_blurry_0, img_blurry_1, img_blurry_2])

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

def show_stack(image, output):
    gauss_stack = gaussian_stack(image, 20, 5)
    gauss_stack = [gauss for gauss in gauss_stack]
    lap_stack = laplacian_stack(image, 20, 5)
    lap_stack = [lapl for lapl in lap_stack]
    gauss_output = np.concatenate(gauss_stack[:], axis=1)
    io.imsave(output + "_gaussianStack.jpg", gauss_output)
    lapl_output = np.concatenate(lap_stack[:], axis=1)
    io.imsave(output + "_laplacianStack.jpg", lapl_output)

img = io.imread("Thanos+CAT.jpg")
show_stack(img, "TC")

### 2.4
def blend(A, B, R):
    R = rgb2gray(R)
    R = R[:,:] > 0.2
    R = np.dstack([R, R, R])
    LA = laplacian_stack(A, 30, 30)
    LB = laplacian_stack(B, 30, 30)
    GR = gaussian_stack(R, 20, 10)
    LS = []
    for i in range(0, len(LA)):
        LS.append(GR[i]*LA[i] + (1 - GR[i])*LB[i])

    LS = sum(LS)

    return LS

img1 = io.imread("sussage.jpg")
img2 = io.imread("akali.jpg")
mask = io.imread("akali_mask.jpg")
result = blend(img1, img2, mask)
io.imsave("Blend4.jpg", result)