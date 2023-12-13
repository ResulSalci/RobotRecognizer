import cv2
import numpy as np

image_org = cv2.imread('Konsol6/src/input/blob-test.png', cv2.IMREAD_GRAYSCALE)

image = cv2.GaussianBlur(image_org, (5, 5), 0)
image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
kernel = np.ones((3, 3), np.uint8)
image = cv2.dilate(image, kernel, iterations=2)
image = cv2.erode(image, kernel, iterations=1)

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 255
params.filterByArea = True
params.minArea = 500
params.maxArea = 1125000
params.filterByCircularity = True
params.minCircularity = 0.85

detector = cv2.SimpleBlobDetector_create(params)

keypoints = detector.detect(image)

image_keypoints = np.zeros_like(image_org)
image_keypoints = cv2.drawKeypoints(image_org, keypoints, image_keypoints, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

blobs_only_image = np.zeros_like(image_org)
blobs_only_image = cv2.drawKeypoints(blobs_only_image, keypoints, blobs_only_image, (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

for kp in keypoints:
    x, y = np.int0(kp.pt)
    cv2.circle(image_keypoints, (x, y), int(kp.size/2), (255, 0, 0), thickness=2)
    cv2.circle(blobs_only_image, (x, y), int(kp.size/2), (255, 255, 255), thickness=2)

cv2.imshow("Original Image", image_org)
cv2.imshow("Keypoints", image_keypoints)
cv2.imshow("Blobs Only", blobs_only_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
