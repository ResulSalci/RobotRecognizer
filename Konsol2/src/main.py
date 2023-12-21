import cv2
import numpy as np
# '/Users/muhammedemincaglar/Documents/GitHub/RobotRecognizer/Konsol2/input/34.jpg'

# Görüntüyü oku
image = cv2.imread('/Users/muhammedemincaglar/Documents/GitHub/RobotRecognizer/Konsol2/input/7.jpg')

# RGB'den HSV'ye dönüştür
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Beyaz renk için renk aralığı belirle
lower_white = np.array([0, 0, 200])
upper_white = np.array([255, 30, 255])

# Siyah renk için renk aralığı belirle
lower_black = np.array([0, 0, 0])
upper_black = np.array([180, 255, 30])

# Renk eşikleme uygula
white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
black_mask = cv2.inRange(hsv_image, lower_black, upper_black)

# Beyaz ve siyah maskelerin kesişimi
intersection_mask = cv2.bitwise_and(white_mask, black_mask)

# Kontur tespiti
contours, _ = cv2.findContours(intersection_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Eğer kontur bulunduysa devam et
if contours:
    # En büyük konturu bul
    max_contour = max(contours, key=cv2.contourArea)

    # Konturu çevrele
    cv2.drawContours(image, [max_contour], -1, (0, 0, 255), 2)

    # Sonuçları göster
    cv2.imshow('Processed Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Kontur bulunamadı.")
