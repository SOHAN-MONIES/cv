import cv2

# Load the image
img = cv2.imread('/home/sohan/coding/python-cv/dog.png')

# Check if the image was loaded properly
if img is None:
    print("Error: Image not found or unable to load.")
else:
    # Resize the image to 50% of its original dimensions
    resized = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

    # Display original and resized images
    cv2.imshow('Original', img)
    cv2.imshow('Resized', resized)

    # Save the resized image
    cv2.imwrite('dog-small.jpg', resized)

    # Wait for a key press and close the image windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
