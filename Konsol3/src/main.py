import cv2
import numpy as np

image = cv2.imread("../input/test_image.jpg")


canny = cv2.Canny(image, 130, 250)

contours, hierarchy = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # Sadece en büyük konturu ayıklıyoruz
    if cv2.contourArea(cnt) > 2000:
        # Konturları çiziyoruz
        cv2.drawContours(image, [cnt], 0, (0, 0, 255), 2)

cv2.imwrite("../output/test_output.jpg", image)


"""

import cv2

image = cv2.imread("../input/test_image.jpg")

img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# apply binary thresholding
ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)

# detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

# draw contours on the original image
image_copy = image.copy()
cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2,
                 lineType=cv2.LINE_AA)

# see the results
cv2.imshow('None approximation', image_copy)
cv2.waitKey(0)
cv2.imwrite('../output/test_output.jpg', image_copy)
cv2.destroyAllWindows()









# visualize the binary image
cv2.imshow('Binary image', thresh)
cv2.waitKey(0)
cv2.imwrite('../output/test_output.jpg', thresh)
cv2.destroyAllWindows()
"""
