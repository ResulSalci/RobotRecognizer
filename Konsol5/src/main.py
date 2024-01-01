import cv2
import os
import numpy as np


def tespit_ve_ciz(img_path, output_path):
    # Resmi oku
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Gri tonlamaya çevir
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gürültüyü azaltmak için Gaussian bulanıklaştırma uygula
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Hough Circle yöntemi ile çemberleri tespit et
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=75, param2=40, minRadius=95, maxRadius=190)

    if circles is not None:
        # Çemberleri çiz
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Çemberin merkezini çiz
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 0), 9)
            # Çembiri çiz
            cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 3)

        # İşlenmiş resmi kaydet
        output_file = os.path.join(output_path, os.path.basename(img_path))
        cv2.imwrite(output_file, img)
        print(f"{output_file} dosyası başarıyla oluşturuldu.")
    else:
        print("Çember bulunamadı.")


# Input klasörü ve output klasörü
input_klasoru = "../input"
output_klasoru = "../output"

# Output klasörünü oluştur
if not os.path.exists(output_klasoru):
    os.makedirs(output_klasoru)

# Input klasöründeki her resmi işle
for file_name in os.listdir(input_klasoru):
    if file_name.endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(input_klasoru, file_name)
        tespit_ve_ciz(img_path, output_klasoru)
