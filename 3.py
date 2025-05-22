import cv2

# Load the image
img = cv2.imread('whiteboard.jpg')

# Check if image was loaded successfully
if img is None:
    print("Error: Unable to load image.")
else:
    # Draw a blue rectangle (BGR: 255, 0, 0), thickness = 3
    cv2.rectangle(img, (50, 50), (200, 150), (255, 0, 0), 3)

    # Draw a filled green circle (BGR: 0, 255, 0)
    cv2.circle(img, (300, 100), 40, (0, 255, 0), -1)

    # Draw a red line (BGR: 0, 0, 255), thickness = 2
    cv2.line(img, (100, 200), (300, 300), (0, 0, 255), 2)

    # Display the image with shapes
    cv2.imshow('Shapes', img)

    # Save the modified image
    cv2.imwrite('shapes_on_whiteboard.jpg', img)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
