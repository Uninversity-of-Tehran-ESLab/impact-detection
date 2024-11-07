import cv2
import numpy as np

def nothing(x):
    pass

# Load the image
image = cv2.imread('your_image.jpg')
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Create a window with trackbars for each HSV component
cv2.namedWindow('Trackbars')
cv2.createTrackbar('Lower H', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('Upper H', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('Lower S', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Upper S', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Lower V', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('Upper V', 'Trackbars', 255, 255, nothing)

while True:
    # Get trackbar positions
    lower_h = cv2.getTrackbarPos('Lower H', 'Trackbars')
    upper_h = cv2.getTrackbarPos('Upper H', 'Trackbars')
    lower_s = cv2.getTrackbarPos('Lower S', 'Trackbars')
    upper_s = cv2.getTrackbarPos('Upper S', 'Trackbars')
    lower_v = cv2.getTrackbarPos('Lower V', 'Trackbars')
    upper_v = cv2.getTrackbarPos('Upper V', 'Trackbars')

    # Set the HSV range
    lower_bound = np.array([lower_h, lower_s, lower_v])
    upper_bound = np.array([upper_h, upper_s, upper_v])

    # Create the mask and apply it to the original image
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)
    filtered_image = cv2.bitwise_and(image, image, mask=mask)

    # Display the result
    cv2.imshow('Filtered Image', filtered_image)
    cv2.imshow('Mask', mask)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
