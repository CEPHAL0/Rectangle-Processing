import cv2
import numpy as np

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

        # Sort the contour points by x-coordinate
        sorted_points = np.sort(approx[:, 0, :], axis=0)

        # Get the top-left and bottom-right points of the rectangle
        top_left = sorted_points[0]
        bottom_right = sorted_points[-1]

        # Calculate the y-coordinate of the horizontal line within the rectangle
        line_y = (top_left[1] + bottom_right[1]) // 2

        # Extract the region of interest (ROI) containing the rectangle
        roi = enhanced_image[top_left[1]:bottom_right[1],
                             top_left[0]:bottom_right[0]]

        # Apply thresholding to the ROI to emphasize the line
        _, roi_thresholded = cv2.threshold(roi, 200, 255, cv2.THRESH_BINARY)

        # Perform Hough Line Transform on the ROI
        lines = cv2.HoughLinesP(
            roi_thresholded, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=10)

        # Check if any lines are detected
        if lines is not None:
            # Iterate over the detected lines
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate the y-coordinate of the line segment midpoint
                line_mid_y = (y1 + y2) // 2

                # Check if the line segment midpoint is close to the desired line y-coordinate
                if abs(line_mid_y - line_y) <= 2:
                    # Draw the line segment on the image
                    cv2.line(image, (top_left[0] + x1, top_left[1] + y1), (top_left[0] + x2, top_left[1] + y2),
                             (0, 0, 255), 2)

                    # Calculate the length of the line segment
                    line_length = abs(x2 - x1)

                    # Print the length of the line segment
                    print(f"Line Length: {line_length}")
                    break  # Exit the loop after finding the first matching line segment

# Display the enhanced image, thresholded image, and the image with the detected rectangle and line
cv2.imshow('Enhanced Image', enhanced_image)
cv2.imshow('Thresholded Image', thresholded_image)
cv2.imshow('Detected Rectangle and Line', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
