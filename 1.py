import cv2
import os

print("Current working directory:", os.getcwd())

img = cv2.imread('dog.png')  # Corrected filename
if img is None:
    print("❌ Error: Image not found or cannot be loaded.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original', img)
cv2.imshow('Grayscale', gray)
cv2.imwrite('dog_gray.png', gray)  # Optional: save as PNG
cv2.waitKey(0)
cv2.destroyAllWindows()
