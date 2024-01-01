import cv2
import numpy as np

# Resmi yükle
image = cv2.imread('../input/106.jpg')

# Gri tonlamaya dönüştür
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gürültüyü azaltmak için Gaussian Blur uygula
blur = cv2.GaussianBlur(gray, (21, 21), 0)

# Kenarları belirleme (Canny Edge Detection)
edges = cv2.Canny(blur, 20, 25)

# Contour (sınırlar) tespiti
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Konturları resmin üzerine çiz
cv2.drawContours(image, contours, -1, (0, 255, 0), 5)

# Sonucu göster
cv2.imshow('Contours', cv2.resize(image,(600,600)))
cv2.waitKey(0)
cv2.destroyAllWindows()



"""
import cv2
import os

def detect_robots(image):
    # Resmi standartlaştırın
    image = cv2.resize(image, (640, 480))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur uygulayın
    blur = cv2.GaussianBlur(image, (5, 5), 1)

    # Kenar tespiti uygulayın
    edges = cv2.Canny(blur, 50, 150)

    # Konturları bulun
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Robotları tespit edin
    robots = []
    for contour in contours:
        # Kontur alanı sınırını kontrol edin
        if cv2.contourArea(contour) > 200:
            # Kontur şeklini kontrol edin
            x, y, w, h = cv2.boundingRect(contour)
            if w > 50 and h > 50:
                robots.append(contour)

    return robots

def main():
    # Input klasörünü açın
    input_dir = "../input"
    input_files = os.listdir(input_dir)

    # Output klasörünü oluşturun
    output_dir = "../output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Her bir resim için
    for input_file in input_files:
        # Resmi yükleyin
        image = cv2.imread(os.path.join(input_dir, input_file))

        # Robotları tespit edin
        robots = detect_robots(image)

        # Robotları çizin
        for robot in robots:
            cv2.drawContours(image, [robot], -1, (0, 255, 0), 2)

        # Resmi output klasörüne kaydedin
        output_file = input_file.replace("input", "output")
        cv2.imwrite(os.path.join(output_dir, output_file), image)

if __name__ == "__main__":
    main()
"""