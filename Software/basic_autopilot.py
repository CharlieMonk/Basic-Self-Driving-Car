import cv2
import numpy as np

img = cv2.imread("/Users/cbmonk/AnacondaProjects/SelfDrivingCar/Test/1.png", 1)

# Define the lower and upper color ranges of the double yellow center lines
center_lines_lower = np.array([0, 124, 210])
center_lines_upper = np.array([160, 234, 255])

# Filter out all parts except the yellow center lines
masked = cv2.inRange(img, center_lines_lower, center_lines_upper)
res = cv2.bitwise_and(img, img, mask=masked)
kernel = np.ones((3,2), np.uint8)
eroded = cv2.morphologyEx(res, cv2.MORPH_OPEN, kernel)
# Show the filtered image
cv2.imshow("Without erosion", res)
cv2.imshow("Modified", eroded)

# Close everything out when any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
