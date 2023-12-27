import cv2
import numpy as np

def anything(a):
    pass

pth = '/Users/muhammedemincaglar/Documents/GitHub/RobotRecognizer/Konsol2/input/1.jpg'
img = cv2.imread(pth)
height, width, _ = img.shape

# (width, height)
if height == 4032:
    img = cv2.resize(img, (600, 802))
else:
    img = cv2.resize(img, (802, 600))

cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 320, 240)
cv2.createTrackbar("Hue Min","Trackbars", 0, 179, anything)
cv2.createTrackbar("Hue Max","Trackbars", 179, 179, anything)
cv2.createTrackbar("Satur Min","Trackbars", 0, 255, anything)
cv2.createTrackbar("Satur Max","Trackbars", 255, 255, anything)
cv2.createTrackbar("Value Min","Trackbars", 0, 255, anything)
cv2.createTrackbar("Value Max","Trackbars", 255, 255, anything)

while True:
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("Hue Min", "Trackbars")
    h_max = cv2.getTrackbarPos("Hue Max", "Trackbars")
    s_min = cv2.getTrackbarPos("Satur Min", "Trackbars")
    s_max = cv2.getTrackbarPos("Satur Max", "Trackbars")
    v_min = cv2.getTrackbarPos("Value Min", "Trackbars")
    v_max = cv2.getTrackbarPos("Value Max", "Trackbars")
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(imgHSV, lower, upper)
    cv2.imshow("Masked", mask)
    cv2.resizeWindow("Masked", 320, 240)
    cv2.imshow("Image", img)
    cv2.resizeWindow("Image", 320, 240)

    key = cv2.waitKey(1) & 0xFF  
    if key == 27 or key == ord('q') or key == ord('a'):
        break

cv2.destroyAllWindows()
