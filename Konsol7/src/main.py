import cv2
import numpy as np
import sys

if len(sys.argv) != 4:
    print("Usage is: python main.py <minArea> <maxArea> <minCircularity>")
    sys.exit(1)

def get_color_from_file(number):
    if(number >= 1 and number <= 2):
        return np.array([[24,14,56], [87,91,175]])
    elif(number >= 3 and number <= 4):
        return np.array([[59,28,114], [89,94,163]])
    elif(number == 5):
        return np.array([[30,25,45], [80,97,138]])
    elif(number == 6):
        return np.array([[27,30,25], [80,160,142]])
    elif(number >= 7 and number <= 8):
        return np.array([[30,28,33], [97,155,160]])
    elif(number >= 9 and number <= 10):
        return np.array([[67,48,58], [97,102,255]])
    elif(number >= 11 and number <= 12):
        return np.array([[41,39,45], [97,87,255]])
    elif(number >= 13 and number <= 14):
        return np.array([[38,53,117], [56,101,255]])
    elif(number >= 15 and number <= 17):
        return np.array([[0,31,45], [64,95,255]])
    elif(number >= 18 and number <= 22):
        return np.array([[0,40,111], [66,85,255]])
    elif(number >= 23 and number <= 24):
        return np.array([[31,20,106], [78,69,255]])
    elif(number >= 25 and number <= 27):
        return np.array([[33,11,116], [80,58,255]])
    elif(number >= 28 and number <= 49):
        return np.array([[75,25,165], [100,55,255]])
    elif(number >= 50 and number <= 62):
        return np.array([[47,25,83], [100,90,255]])
    elif(number >= 63 and number <= 69):
        return np.array([[42,5,83], [107,75,170]])
    elif(number >= 70 and number <= 79):
        return np.array([[50,0,160], [179,53,255]])
    elif(number >= 80 and number <= 89):
        return np.array([[20,0,11], [132,130,74]])
    elif(number >= 90 and number <= 94):
        return np.array([[65,0,45], [131,124,171]])
    elif(number >= 95 and number <= 99):
        return np.array([[100,0,53], [141,120,181]])
    elif(number >= 100 and number <= 108):
        return np.array([[100,20,63], [122,125,153]])
    else:
        return None

#108
for i in range(108):

    if i == 15:
        continue

    image_org = cv2.imread(f'Konsol7/input/{i+1}.jpg')

    # (width, height)
    image_org = cv2.resize(image_org, (600, 451))

    image = cv2.cvtColor(image_org, cv2.COLOR_BGR2HSV)

    image = cv2.GaussianBlur(image, (9, 9), 0)
    #image = cv2.bitwise_not(image)
    # 192, 193
    # image = cv2.threshold(image, 177, 180, cv2.THRESH_BINARY)[1]

    color_range = get_color_from_file(i+1)

    image = cv2.inRange(image, color_range[0], color_range[1])
    image = cv2.bitwise_not(image)
    # kernel = np.ones((3, 3), np.uint8)
    image = cv2.dilate(image, (21, 21), iterations=4)
    #image = cv2.erode(image, (3,3), iterations=1)

    params = cv2.SimpleBlobDetector_Params()
    # params.minThreshold = 10
    # params.maxThreshold = 255
    params.filterByArea = True
    params.minArea = int(sys.argv[1])
    params.maxArea = int(sys.argv[2])

    params.filterByCircularity = True
    params.minCircularity = float(sys.argv[3])


    detector = cv2.SimpleBlobDetector_create(params)

    keypoints = detector.detect(image)

    image_keypoints = np.zeros_like(image_org)
    image_keypoints = cv2.drawKeypoints(image, keypoints, image_keypoints, (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    blobs_only_image = np.zeros_like(image_org)
    blobs_only_image = cv2.drawKeypoints(blobs_only_image, keypoints, blobs_only_image, (255, 255, 255),
                                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imwrite(f"Konsol7/output/{i + 1}_output.jpg", image_keypoints)

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