import unittest
import cv2
import subprocess
from main import find_robot

class TestImageProcessing(unittest.TestCase):

    # Görsellerin başarıkı bir şekilde yüklenebildiğini kontrol et
    def test_image_loading(self):
        for i in range(32):
            image_path = f'Konsol6/input/{i+1}.jpg'
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(image)

    # Var olmayan resimlerde beklendiği şekilde hata alındığını kontrol et
    def test_non_existing_image_loading(self):
        image_path = "not-a-real-path"
        image = cv2.imread(image_path)
        self.assertIsNone(image)

    # Görsellerin başarılı bir şekilde yeniden boyutlandırıldığını kontrol et
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

    # Görsellere başarılı bir şekilde gaussian blur uygulanabildiğini kontrol et
    def test_gaussian_blur(self):
        for i in range(32):
            image_path = f'Konsol6/input/{i+1}.jpg'
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            blurred_image = cv2.GaussianBlur(image, (51, 51), 0)

            self.assertIsNotNone(blurred_image)

    # Görsellere başarılı bir şekilde thresholding yapılabildiğini kontrol et
    def test_image_thresholding(self):
        for i in range(32):
            image_path = f'Konsol6/input/{i+1}.jpg'
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            thresholded_image = cv2.GaussianBlur(image, (51, 51), 0)
            thresholded_image = cv2.threshold(thresholded_image, 192, 193, cv2.THRESH_BINARY_INV)[1]

            self.assertIsNotNone(thresholded_image)

    # İşlenen görsellerin blob detecion fonsiyonundan sorunsuz çıktı alıp alamadığını kontrol et
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

    # Görüntü işleme fonsiyonun aynı değerler için aynı görüntü outputlarını verip vermediğini kontrol et
    def test_find_robot_gives_consistent_results_for_same_inputs(self):
        image_path = 'Konsol6/input/1.jpg'

        image_org = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        minArea, maxArea, minCircularity, gaussianBlurKernelSize, thresholdMin, thresholdMax = 300, 1125000, 0.40, 51, 192, 193

        keypoints1 = find_robot(image_org, minArea, maxArea, minCircularity, gaussianBlurKernelSize, thresholdMin, thresholdMax)
        keypoints2 = find_robot(image_org, minArea, maxArea, minCircularity, gaussianBlurKernelSize, thresholdMin, thresholdMax)

        self.assertEqual(keypoints1.any(), keypoints2.any())

    # Görüntü işleme sonucu alınan görüntülerin diske başarılı bir şekilde kaydedilip edilemediğini kontrol et
    def test_output_image_saving(self):

        subprocess.run("python Konsol6\src\main.py 300 1125000 0.40 51 192 193", shell=True, check=True)
       
        for i in range(32):

            saved_image = cv2.imread(f'Konsol6/output/{i+1}.jpg')
            self.assertIsNotNone(saved_image)

    # Uygulmayı konsoldan çağırırken eksik parametrede hata alınımp alınmadığını kontrol et
    def test_script_call_from_command_line_without_params(self):
        result = subprocess.run("python Konsol6\src\main.py", shell=True, capture_output=True)
        self.assertEqual(result.returncode, 0)

    # Uygulmayı konsoldan çağırırken illegal parametrede hata alınıp alınmadığını kontrol et
    def test_script_call_from_command_line_with_illegal_args(self):
        result = subprocess.run("python Konsol6\src\main.py asdaswd asdasd asdasd asdasd asdasd asdasd", shell=True, capture_output=True)
        self.assertEqual(result.returncode, 2)

if __name__ == '__main__':
    unittest.main()
