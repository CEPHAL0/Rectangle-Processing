import cv2
import numpy as np
import os

# Example image filenames
image_filenames = ['rect1.png', 'rect2.png',
                   'rect3.png', 'rect4.png']

# Load and process each image
line_lengths = []
for filename in image_filenames:
    # Load the image
    image = cv2.imread(os.path.join("original_images",filename))

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Perform line detection using HoughLines method
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    # Calculate the length of the longest line
    longest_line_length = 0
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        if line_length > longest_line_length:
            longest_line_length = line_length

    # Append the longest line length to the list
    line_lengths.append(longest_line_length)

# Sort the image filenames based on line lengths in ascending order
sorted_image_filenames = [x for _, x in sorted(
    zip(line_lengths, image_filenames))]

# Print the sorted image filenames and line lengths
for i, filename in enumerate(sorted_image_filenames):
    print(f"Rectangle {i+1}: {filename} (Length: {line_lengths[i]})")
