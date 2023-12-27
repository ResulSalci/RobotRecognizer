import cv2
import numpy as np
import sys

# minArea = 300, maxArea = 1125000, minCircularity = 0.40
if len(sys.argv) != 4:
    print("Usage is: python main.py <minArea> <maxArea> <minCircularity>")
    sys.exit(1)

def analyse_images(minArea, maxArea, minCircularity):
    for i in range(32):
        image_org = cv2.imread(f'Konsol6/input/{i+1}.jpg', cv2.IMREAD_GRAYSCALE)

        height = image_org.shape[0]
        width = image_org.shape[1]

        #(width, height)
        if height == 4032:
            image_org = cv2.resize(image_org, (600, 802))

        else:
            image_org = cv2.resize(image_org, (802, 600))

        image = cv2.GaussianBlur(image_org, (51, 51), 0)
        #192, 193
        image = cv2.threshold(image, 192, 193, cv2.THRESH_BINARY_INV)[1]
        #kernel = np.ones((3, 3), np.uint8)
        #image = cv2.dilate(image, kernel, iterations=2)
        #image = cv2.erode(image, kernel, iterations=1)

        params = cv2.SimpleBlobDetector_Params()
        #params.minThreshold = 10
        #params.maxThreshold = 255
        params.filterByArea = True
        params.minArea = minArea
        params.maxArea = maxArea
        params.filterByCircularity = True
        params.minCircularity = minCircularity

        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(image)

        image_keypoints = np.zeros_like(image_org)
        image_keypoints = cv2.drawKeypoints(image_org, keypoints, image_keypoints, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        blobs_only_image = np.zeros_like(image_org)
        blobs_only_image = cv2.drawKeypoints(blobs_only_image, keypoints, blobs_only_image, (255, 255, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imwrite(f"Konsol6/output/{i+1}.jpg", image_keypoints)

        """
        for kp in keypoints:
            x, y = np.int0(kp.pt)
            cv2.circle(image_keypoints, (x, y), int(kp.size/2), (255, 0, 0), thickness=2)
            cv2.circle(blobs_only_image, (x, y), int(kp.size/2), (255, 255, 255), thickness=2)
        
        cv2.imshow("Original Image", image_org)
        cv2.imshow("Keypoints", image_keypoints)
        cv2.imshow("Blobs Only", blobs_only_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

analyse_images(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))