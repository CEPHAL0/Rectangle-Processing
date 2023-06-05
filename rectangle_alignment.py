import cv2
import numpy as np

# Load the image
image = cv2.imread('original_images/rect2.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding
_, thresholded = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(
    thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the largest rectangle contour
largest_contour = max(contours, key=cv2.contourArea)

# Approximate the largest contour to a polygon
approx = cv2.approxPolyDP(largest_contour, 0.02 *
                          cv2.arcLength(largest_contour, True), True)

# Get the bounding rectangle of the polygon
x, y, w, h = cv2.boundingRect(approx)

# Calculate the coordinates of the rectangle corners
rect_points = np.array(
    [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32)

# Sort the rectangle points in clockwise order starting from the top-left corner
rect_points = rect_points[np.argsort(rect_points[:, 0])]

# Calculate the width and height of the rectangle
width = np.linalg.norm(rect_points[0] - rect_points[1])
height = np.linalg.norm(rect_points[1] - rect_points[2])

# Set the destination points for the transformation
dst_points = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1]])

# Perform the perspective transformation
M = cv2.getAffineTransform(rect_points[:3], dst_points)
aligned_image = cv2.warpAffine(image, M, (int(width), int(height)))

# Display the aligned image
cv2.imshow('Aligned Image', aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
