import unittest
import numpy as np
import cv2
import subprocess
import os



class TestGetColorFromFileAndGetFileName(unittest.TestCase):
    # Countour bulma testi
    def test_thresholding_and_contour_finding(self):
        # Test için kullanılacak resmin oluşturulması
        image = np.zeros((100, 100, 3), dtype=np.uint8)  # 100x100 boyutunda siyah bir resim
        color = (0, 255, 0)  # Yeşil renk
        cv2.rectangle(image, (10, 10), (30, 30), color, thickness=cv2.FILLED)  # Yeşil dikdörtgen ekleniyor

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_bound = np.array([40, 50, 50]) 
        upper_bound = np.array([80, 255, 255])
        thresholded_image = cv2.inRange(hsv_image, lower_bound, upper_bound)
        contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        expected_contour_count = 1
        expected_contour_area = 400  # 20x20 boyutundaki yeşil dikdörtgenin alanı
        self.assertEqual(len(contours), expected_contour_count)
        self.assertEqual(cv2.contourArea(contours[0]), expected_contour_area)

    def test_robot_detection_and_marking(self):
        # Test için resim oluşturuyorum
        image = cv2.imread("/Users/muhammedemincaglar/Documents/GitHub/RobotRecognizer/Konsol1/input/1.jpg")
        color_range = [np.array([0, 0, 200]), np.array([255, 30, 255])]
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_image, color_range[0], color_range[1])
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]

        # Bulunduğu bütün nesneleri robot diye işaretle
        for cnt in filtered_contours:
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

        # Resmi göster
        cv2.imshow("Robotlar", image)
        cv2.waitKey(0)
    # RGB rengin HSV döndürüp döndürmediğinin kontrolu
    def test_rgb_to_hsv_conversion(self):
        rgb_color = np.array([0, 255, 0], dtype=np.uint8)
        # RGB'den HSV'ye dönüştürme
        hsv_color = cv2.cvtColor(np.uint8([[rgb_color]]), cv2.COLOR_RGB2HSV)[0][0]
        # Beklenen HSV renk
        expected_hsv_color = np.array([60, 255, 255], dtype=np.uint8)
        error_margin = 1
        # Her bir kanalın karşılaştırılması
        for i in range(3):
            self.assertLessEqual(abs(hsv_color[i] - expected_hsv_color[i]), error_margin)
    # resize fonksiyonu test
    def test_image_resize(self):
        input_image = np.zeros((100, 150, 3), dtype=np.uint8)  # 100x150 boyutunda siyah bir resim
        resized_image = cv2.resize(input_image, (50, 75))
        expected_height = 75
        expected_width = 50
        actual_height, actual_width, _ = resized_image.shape
        self.assertEqual(actual_height, expected_height)
        self.assertEqual(actual_width, expected_width)
    
    def test_non_existing_image_loading(self):
        image_path = "not-a-real-path"
        image = cv2.imread(image_path)
        self.assertIsNone(image)
    
    def test_image_loading(self):
        for i in range(32):
            image_path = f'/Users/muhammedemincaglar/Documents/GitHub/RobotRecognizer/Konsol1/input/{i+1}.jpg'
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(image)
    def test_wrong_color_range(self):
        # Yanlış renk aralığı belirleme testi
        image = np.zeros((100, 100, 3), dtype=np.uint8)  # Siyah bir resim
        color_range = [np.array([255, 255, 255]), np.array([255, 255, 255])]  # Yanlış renk aralığı
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv_image, color_range[0], color_range[1])
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.assertEqual(len(contours), 0)  # Hiç kontur bulunmamalı

    def test_corrupt_image_loading(self):
        # Bozuk görüntü yükleme testi
        image_path = "/Users/muhammedemincaglar/Documents/GitHub/RobotRecognizer/Konsol1/input/33.jpg"  # Var olmayan bir görüntü yolu
        image = cv2.imread(image_path)
        self.assertIsNone(image)  # None dönmeli

    def test_corrupt_image_resize(self):
        # Bozuk boyut değiştirme testi
        input_image = np.zeros((100, 150, 3), dtype=np.uint8)  # 100x150 boyutunda siyah bir resim
        resized_image = cv2.resize(input_image, (50, 75))  # Pozitif boyut belirtildi
        self.assertIsNotNone(resized_image)  # None dönmeli

    def test_save_image(self):
        # Görüntüyü kaydetme testi
        image = cv2.imread("/Users/muhammedemincaglar/Documents/GitHub/RobotRecognizer/Konsol1/input/1.jpg")
        cv2.imshow("Test Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        saved_image_path = "/Users/muhammedemincaglar/Documents/GitHub/RobotRecognizer/Konsol1/output/saved_image.jpg"
        cv2.imwrite(saved_image_path, image)
        self.assertTrue(os.path.exists(saved_image_path))  # Dosya kaydedildiyse True dönmeli
  
if __name__ == '__main__':
    unittest.main()