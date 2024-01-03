import unittest
import cv2
import numpy as np
import subprocess
from main import find_robot, get_color_from_file


class TestImageProcessing(unittest.TestCase):

    # Görsellerin başarıkı bir şekilde yüklenebildiğini kontrol et
    def test_image_loading(self):
        image_path = 'Konsol7/input/22.jpg'
        image = cv2.imread(image_path)
        self.assertIsNotNone(image)

    # Var olmayan resimlerde beklendiği şekilde hata alındığını kontrol et
    def test_non_existing_image_loading(self):
        image_path = "not-a-real-path"
        image = cv2.imread(image_path)
        self.assertIsNone(image)

    # Görsellerin başarılı bir şekilde yeniden boyutlandırıldığını kontrol et
    def test_image_resizing(self):
        image_path = 'Konsol7/input/22.jpg'
        image = cv2.imread(image_path)

        resized_image = cv2.resize(image, (600, 451))

        self.assertEqual(resized_image.shape[:2], (451, 600))

    # Görselin BGR uzayından HSV dönüştürülüp dönüştürülemediğini kontrol et
    def test_image_color_from_bgr_to_hsv(self):

        image_path = 'Konsol7/input/22.jpg'
        image = cv2.imread(image_path)

        color_extracted_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        self.assertIsNotNone(color_extracted_image)

    # Görsellere başarılı bir şekilde gaussian blur uygulanabildiğini kontrol et
    def test_gaussian_blur(self):

        image_path = 'Konsol7/input/22.jpg'
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        blurred_image = cv2.GaussianBlur(image, (9, 9), 0)

        self.assertIsNotNone(blurred_image)

    # İşlenen görsellerin blob detecion fonsiyonundan sorunsuz çıktı alıp alamadığını kontrol et
    def test_blob_detection(self):

        image_org = cv2.imread('Konsol7/input/22.jpg')

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

    # Görüntü işleme fonsiyonun aynı değerler için aynı görüntü outputlarını verip vermediğini kontrol et
    def test_find_robot_gives_consistent_results_for_same_inputs(self):
        image_path = 'Konsol7/input/1.jpg'

        image_org = cv2.imread(image_path)

        minArea, maxArea, gaussianBlurKernelSize = 300, 1125000, 9

        keypoints1 = find_robot(image_org, 0, minArea, maxArea, gaussianBlurKernelSize)
        keypoints2 = find_robot(image_org, 0, minArea, maxArea, gaussianBlurKernelSize)

        self.assertEqual(keypoints1.any(), keypoints2.any())

    # Görüntü işleme sonucu alınan görüntülerin diske başarılı bir şekilde kaydedilip edilemediğini kontrol et
    def test_output_image_saving(self):

        subprocess.run("python Konsol7\src\main.py 300 1125000 9", shell=True, check=True)
       
        for i in range(108):

            if i == 15:
                continue

            saved_image = cv2.imread(f'Konsol7/output/{i+1}.jpg')
            self.assertIsNotNone(saved_image)

    # Uygulmayı konsoldan çağırırken eksik parametrede hata alınımp alınmadığını kontrol et
    def test_script_call_from_command_line_without_params(self):
        result = subprocess.run("python Konsol7\src\main.py", shell=True, capture_output=True)
        self.assertEqual(result.returncode, 1)

    # Uygulmayı konsoldan çağırırken illegal parametrede hata alınıp alınmadığını kontrol et
    def test_script_call_from_command_line_with_illegal_args(self):
        result = subprocess.run("python Konsol7\src\main.py asdaswd asdasd asdas", shell=True, capture_output=True)
        self.assertEqual(result.returncode, 1)


if __name__ == '__main__':
    unittest.main()
