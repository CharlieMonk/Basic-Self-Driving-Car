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
    lower_canny = 200
    upper_canny = 300

    # Use canny to detect edges
    edges = cv2.Canny(closed, lower_canny, upper_canny)
    # Fit lines to the canny edges
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=40, maxLineGap=10)

    # If thresholding didn't work, use the original image
    if (lines == None):
        edges = cv2.Canny(img, lower_canny, upper_canny)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=40, maxLineGap=10)
    for edge in edges:
        img2, contour, hierarchy = cv2.findContours(edge, 1, 2)
        cv2.imshow("Test", img2)
        contour_area = 1#cv2.contourArea(contour)
        if(contour_area > 30):
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = float(contour_area)/hull_area
            if(solidity>0.7):
                x1, y1, x2, y2 = cv2.HoughLinesP(edge, 1, np.pi/180, 100, minLineLength=10, maxLineGap=10)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)
    # Draw the lines (remove later)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)


    # Draw rotated rectangles around the contours
    img2, contours, hierarchy = cv2.findContours(edges, 1, 2)
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if(contour_area > 30):
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = float(contour_area)/hull_area
            if (solidity<0.8) and (solidity>0.05):
                rect = cv2.minAreaRect(contour)
                box = np.int0(cv2.boxPoints(rect))
                cv2.drawContours(img, [box], 0, (0,0,255))
                print(solidity)
    cv2.imshow("edges", img2)
    return img

# Run findRoadLines on a test image
linesFound = findRoadLines("/Users/cbmonk/AnacondaProjects/SelfDrivingCar/Test/8.png")
cv2.imshow("Lines detected", linesFound)

# Close everything out when any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
