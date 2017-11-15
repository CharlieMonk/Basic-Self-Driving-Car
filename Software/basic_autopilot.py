import cv2
import numpy as np

img = cv2.imread("/Users/cbmonk/AnacondaProjects/SelfDrivingCar/Test/4.png", 1)

# Define the lower and upper color ranges of the double yellow center lines
center_lines_lower = np.array([0, 124, 210])
center_lines_upper = np.array([160, 234, 255])

# Filter out all parts except the yellow center lines
res = cv2.inRange(img, center_lines_lower, center_lines_upper)
#res = cv2.bitwise_and(img, img, mask=masked) # Enable this to enable color in masked image
kernel = np.ones((3,3), np.uint8)
eroded = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kernel)
lower_canny = 280
upper_canny = 400
# Image thresholding
edges = cv2.Canny(eroded, lower_canny, upper_canny)
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=20,maxLineGap=10)
if(lines == None):
    edges = cv2.Canny(img, lower_canny, upper_canny)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=20,maxLineGap=10)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0,255,0), thickness=10)

# Show the filtered image
cv2.imshow("Road Features Detection", img)
#cv2.imshow("Modified", edges)

# Close everything out when any key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
