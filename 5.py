
import cv2
# Load the image
img = cv2.imread('dog-small.jpg')
# Apply a basic blur
blur = cv2.GaussianBlur(img, (15, 15), 0)
# Show original and blurred images
cv2.imshow('Original', img)
cv2.imshow('Blurred', blur)
# Save the blurred image
cv2.imwrite('market_blurred.jpg', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()