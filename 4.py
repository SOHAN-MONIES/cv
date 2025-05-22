import cv2

# Load the image
img = cv2.imread('dog-small.jpg')

# Check if image was loaded successfully
if img is None:
    print("Error: Unable to load image.")
else:
    # Flip horizontally (left-right)
    flip_h = cv2.flip(img, 1)

    # Flip vertically (top-bottom)
    flip_v = cv2.flip(img, 0)

    # Show all images
    cv2.imshow('Original', img)
    cv2.imshow('Flipped Horizontally', flip_h)
    cv2.imshow('Flipped Vertically', flip_v)

    # Save the flipped images
    cv2.imwrite('dog_flip_h.jpg', flip_h)
    cv2.imwrite('dog_flip_v.jpg', flip_v)

    # Wait for a key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
