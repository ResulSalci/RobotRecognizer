import unittest
import cv2
import numpy as np
import subprocess


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


class TestImageProcessing(unittest.TestCase):


    def test_image_loading(self):
        image_path = f'Konsol7/input/22.jpg'
        image = cv2.imread(image_path)
        self.assertIsNotNone(image)

    def test_non_existing_image_loading(self):
        image_path = "not-a-real-path"
        image = cv2.imread(image_path)
        self.assertIsNone(image)

    def test_image_resizing(self):
        image_path = f'Konsol7/input/22.jpg'
        image = cv2.imread(image_path)

        resized_image = cv2.resize(image, (600, 451))

        self.assertEqual(resized_image.shape[:2], (451, 600))

    def test_image_color_extraction(self):

        image_path = f'Konsol7/input/22.jpg'
        image = cv2.imread(image_path)

        color_extracted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        self.assertIsNotNone(color_extracted_image)

    def test_blob_detection(self):

        image_org = cv2.imread(f'Konsol7/input/22.jpg')

        image_org = cv2.resize(image_org, (600, 451))

        image = cv2.cvtColor(image_org, cv2.COLOR_BGR2HSV)

        image = cv2.GaussianBlur(image, (9, 9), 0)


        color_range = get_color_from_file(22)

        image = cv2.inRange(image, color_range[0], color_range[1])
        image = cv2.bitwise_not(image)

        image = cv2.dilate(image, (21, 21), iterations=4)

        params = cv2.SimpleBlobDetector_Params()

        params.filterByArea = True
        params.minArea = 100
        params.maxArea = 1000000


        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(image)

        self.assertIsNotNone(keypoints)

    def test_output_image_saving(self):

        subprocess.run("python Konsol7\src\main.py 300 1125000 0.40", shell=True, check=True)
       
        for i in range(108):

            if i == 15:
                continue

            saved_image = cv2.imread(f'Konsol7/output/{i+1}.jpg')
            self.assertIsNotNone(saved_image)


if __name__ == '__main__':
    unittest.main()
