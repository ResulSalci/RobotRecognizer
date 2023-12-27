import unittest
import cv2
import numpy as np

class TestImageProcessing(unittest.TestCase):

    def test_image_loading(self):
        for i in range(32):
            image_path = f'Konsol6/input/{i+1}.jpg'
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(image)

    def test_image_resizing(self):
        for i in range(32):
            image_path = f'Konsol6/input/{i+1}.jpg'
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            height = image.shape[0]

            resized_image = None

            if height == 4032:
                resized_image = cv2.resize(image, (600, 802))
                self.assertEqual(resized_image.shape[:2], (802, 600))
            else:
                resized_image = cv2.resize(image, (802, 600))
                self.assertEqual(resized_image.shape[:2], (600, 802))


    def test_image_thresholding(self):
        for i in range(32):
            image_path = f'Konsol6/input/{i+1}.jpg'
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            thresholded_image = cv2.GaussianBlur(image, (51, 51), 0)
            thresholded_image = cv2.threshold(thresholded_image, 192, 193, cv2.THRESH_BINARY_INV)[1]

            self.assertIsNotNone(thresholded_image)

    def test_blob_detection(self):
        for i in range(32):
            image_path = f'Konsol6/input/{i+1}.jpg'
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 300
            params.maxArea = 1125000
            params.filterByCircularity = True
            params.minCircularity = 0.40

            detector = cv2.SimpleBlobDetector_create(params)

            keypoints = detector.detect(image)

            self.assertIsNotNone(keypoints)

    def test_output_image_saving(self):
        for i in range(32):
            image_path = f'Konsol6/input/{i+1}.jpg'
            image_org = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            height = image_org.shape[0]
            width = image_org.shape[1]

            if height == 4032:
                image_org = cv2.resize(image_org, (600, 802))
            else:
                image_org = cv2.resize(image_org, (802, 600))

            image = cv2.GaussianBlur(image_org, (51, 51), 0)
            image = cv2.threshold(image, 192, 193, cv2.THRESH_BINARY_INV)[1]

            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 300
            params.maxArea = 1125000
            params.filterByCircularity = True
            params.minCircularity = 0.40

            detector = cv2.SimpleBlobDetector_create(params)

            keypoints = detector.detect(image)

            image_keypoints = np.zeros_like(image_org)
            image_keypoints = cv2.drawKeypoints(image_org, keypoints, image_keypoints, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

            cv2.imwrite(f'Konsol6/output/{i+1}.jpg', image_keypoints)

            saved_image = cv2.imread(f'Konsol6/output/{i+1}.jpg')
            self.assertIsNotNone(saved_image)

if __name__ == '__main__':
    unittest.main()
