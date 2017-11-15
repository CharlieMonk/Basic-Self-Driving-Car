import cv2
import numpy as np

def findRoadLines(image_path):
    img = cv2.imread(image_path, 1)

    # Lower and upper ranges for yellow center lines and white lane lines
    center_lines_lower = np.array([0, 104, 210])
    center_lines_upper = np.array([255, 255, 255]) #([160, 234, 255])

    # Mask image using center_lines_lower and center_lines_upper
    res = cv2.inRange(img, center_lines_lower, center_lines_upper)

    # Morph closing on image
    kernel = np.ones((3,3), np.uint8)
    closed = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
    # Lower and upper canny thresholds
    lower_canny = 280
    upper_canny = 400
    # Use canny to detect edges
    edges = cv2.Canny(closed, lower_canny, upper_canny)
    # Fit lines to the canny edges
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=20, maxLineGap=10)
    # If thresholding didn't work, use the original image
    if(lines == None):
        edges = cv2.Canny(img, lower_canny, upper_canny)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=20, maxLineGap=10)
    # Draw the lines
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)

    return img

# Run findRoadLines on a test image
linesFound = findRoadLines("/Users/cbmonk/AnacondaProjects/SelfDrivingCar/Test/1.png")
cv2.imshow("Lines detected", linesFound)

# Close everything out when any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
