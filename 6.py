import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from skimage.restoration import wiener

# Load image in grayscale
image = cv2.imread('dog-small.jpg', cv2.IMREAD_GRAYSCALE)

# ---------- Enhancement Functions ----------

def contrast_stretch(img):
    min_val, max_val = np.min(img), np.max(img)
    stretched = (img - min_val) * (255 / (max_val - min_val))
    return np.uint8(stretched)

def histogram_equalization(img):
    return cv2.equalizeHist(img)

def log_transform(img):
    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log(1 + img.astype(np.float32))
    return np.uint8(log_img)

def gamma_transform(img, gamma=0.5):
    img_normalized = img / 255.0
    gamma_corrected = np.power(img_normalized, gamma)
    return np.uint8(gamma_corrected * 255)

# ---------- Smoothing Filters ----------

def average_filter(img, ksize=5):
    return cv2.blur(img, (ksize, ksize))

def gaussian_filter(img, ksize=5, sigma=1):
    return cv2.GaussianBlur(img, (ksize, ksize), sigma)

# ---------- Sharpening ----------

def laplacian_sharpening(img):
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    sharpened = img - 0.5 * laplacian
    return np.uint8(np.clip(sharpened, 0, 255))

def unsharp_mask(img):
    blurred = cv2.GaussianBlur(img, (9, 9), 10.0)
    sharp = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return sharp

# ---------- Motion Blur + Restoration ----------

def motion_blur_kernel(size=15):
    kernel = np.zeros((size, size))
    kernel[int((size-1)/2), :] = np.ones(size)
    return kernel / size

def inverse_filtering(blurred, kernel):
    kernel_padded = np.zeros_like(blurred)
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel

    blurred_fft = np.fft.fft2(blurred)
    kernel_fft = np.fft.fft2(kernel_padded)
    restored_fft = blurred_fft / (kernel_fft + 1e-8)
    restored = np.abs(np.fft.ifft2(restored_fft))
    return np.uint8(np.clip(restored, 0, 255))

def wiener_filtering(blurred, kernel, K=0.01):
    return wiener(blurred, kernel, K)

# ---------- Apply All ----------

contrast_img     = contrast_stretch(image)
hist_eq_img      = histogram_equalization(image)
log_img          = log_transform(image)
gamma_img        = gamma_transform(image, gamma=0.4)
avg_img          = average_filter(image)
gauss_img        = gaussian_filter(image)
laplace_img      = laplacian_sharpening(image)
unsharp_img      = unsharp_mask(image)

# Create motion blur
kernel = motion_blur_kernel()
blurred = convolve2d(image, kernel, 'same')

# Deblur
restored_inv    = inverse_filtering(blurred, kernel)
restored_wiener = wiener_filtering(blurred, kernel)

# ---------- Display Function ----------

def show(title, img):
    plt.figure()
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.axis('off')

# ---------- Show All Images ----------

show("Original", image)
show("Contrast Stretching", contrast_img)
show("Histogram Equalization", hist_eq_img)
show("Log Transform", log_img)
show("Gamma Transform", gamma_img)
show("Average Filter", avg_img)
show("Gaussian Filter", gauss_img)
show("Laplacian Sharpening", laplace_img)
show("Unsharp Masking", unsharp_img)
show("Motion Blurred", blurred)
show("Restored (Inverse Filter)", restored_inv)
show("Restored (Wiener Filter)", restored_wiener)

plt.show()
