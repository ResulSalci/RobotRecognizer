import cv2
import numpy as np

img = cv2.imread('Konsol5/src/input/test.png', cv2.IMREAD_GRAYSCALE)
  
img = cv2.GaussianBlur(img, (3, 3), 0) 
  
detected_circles = cv2.HoughCircles(img,  
                   cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
               param2 = 40, minRadius = 1, maxRadius = 4000) 
  
if detected_circles is not None: 
  
    detected_circles = np.uint16(np.around(detected_circles)) 
  
    for pt in detected_circles[0, :]: 
        a, b, r = pt[0], pt[1], pt[2] 
  
        cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
  
        cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
        cv2.imshow("Detected Circle", img) 
        cv2.waitKey(0)