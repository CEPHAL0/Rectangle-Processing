import cv2
import numpy as np
import os

# Load the image
image = cv2.imread('images (2).png')

# Apply image enhancement techniques (e.g., histogram equalization)
enhanced_image = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

# Apply adaptive thresholding
thresholded_image = cv2.adaptiveThreshold(
    enhanced_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

# Perform edge detection
edges = cv2.Canny(thresholded_image, 50, 150)

# Find contours
contours, hierarchy = cv2.findContours(
    edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Iterate over contours
for contour in contours:
    # Approximate the contour to a polygon
    approx = cv2.approxPolyDP(
        contour, 0.02 * cv2.arcLength(contour, True), True)

    # Check if the contour has four corners (a rectangle)
    if len(approx) == 4:
        # Draw the rectangle on the image
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

        # Draw the line inside the rectangle
        cv2.line(image, tuple(approx[0][0]),
                 tuple(approx[2][0]), (0, 0, 255), 2)
        
    

# Display the enhanced image, thresholded image, and the image with the detected rectangle and line
cv2.imshow('Enhanced Image', enhanced_image)
cv2.imshow('Thresholded Image', thresholded_image)
cv2.imshow('Detected Rectangle and Line', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
