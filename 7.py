import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in BGR
image = cv2.imread('dog-small.jpg')

# Convert to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert to grayscale for thresholding and edge detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Simple binary threshold
_, binary_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Adaptive mean threshold
adaptive_thresh = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Otsu's thresholding
_, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# Morphological operations kernel
kernel = np.ones((5, 5), np.uint8)

# Dilation
dilated = cv2.dilate(binary_thresh, kernel, iterations=1)

# Erosion
eroded = cv2.erode(binary_thresh, kernel, iterations=1)

# Opening (erosion followed by dilation)
opening = cv2.morphologyEx(binary_thresh, cv2.MORPH_OPEN, kernel)

# Closing (dilation followed by erosion)
closing = cv2.morphologyEx(binary_thresh, cv2.MORPH_CLOSE, kernel)

# Convert to HSV for color segmentation
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define range for red color segmentation (lower red)
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
mask_red = cv2.inRange(hsv, lower_red, upper_red)

# Mask the red regions on the RGB image
red_segment = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_red)

# Split channels (BGR) and merge as RGB manually
b, g, r = cv2.split(image)
merged = cv2.merge([r, g, b])

# Titles and images list
titles = [
    "Original Image", "1.1 Binary Threshold", "1.2 Adaptive Threshold",
    "1.3 Otsu Threshold", "1.4 Canny Edges", "2.1 Dilation",
    "2.2 Erosion", "2.3 Opening", "2.4 Closing",
    "3.1 HSV Image", "3.2 Red Color Segmentation", "3.3 RGB Merged from Channels"
]

images = [
    image_rgb, binary_thresh, adaptive_thresh, otsu_thresh, edges, dilated,
    eroded, opening, closing, hsv, red_segment, merged
]

# Create subplots - here 3 rows x 4 columns (adjust based on number of images)
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

for idx, ax in enumerate(axes):
    if idx < len(images):
        img = images[idx]
        # Show grayscale images with cmap='gray'
        if len(img.shape) == 2:
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.set_title(titles[idx])
        ax.axis('off')
    else:
        ax.axis('off')  # Hide any extra subplots

plt.tight_layout()
plt.show()
