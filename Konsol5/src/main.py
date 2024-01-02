import cv2
import os
import numpy as np
import argparse


def tespit_ve_ciz(args):
    # Input klasörü ve output klasörü
    input_folder = "../input"
    output_folder = "../output"

    # Input klasöründeki her resmi işle
    for image_file in os.listdir(input_folder):
        # Görüntüyü oku
        img = cv2.imread(os.path.join(input_folder, image_file), cv2.IMREAD_COLOR)

        # Gri tonlamaya çevir
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Gürültüyü azaltmak için Gaussian bulanıklaştırma uygula
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Hough Circle yöntemi ile çemberleri tespit et
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=args.minDist,
                                   param1=args.param1, param2=args.param2, minRadius=args.minRadius, maxRadius=args.maxRadius)

        if circles is not None:
            # Çemberleri çiz
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Çemberin merkezini çiz
                cv2.circle(img, (i[0], i[1]), 2, (0, 0, 0), 9)
                # Çemberi çiz
                cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 3)

            # İşlenmiş resmi kaydet
            cv2.imwrite(os.path.join(output_folder, image_file), img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Çember tespit etme parametreleri')
    parser.add_argument('minDist', type=int, default=100, nargs='?')
    parser.add_argument('param1', type=int, default=75, nargs='?')
    parser.add_argument('param2', type=int, default=40, nargs='?')
    parser.add_argument('minRadius', type=int, default=95, nargs='?')
    parser.add_argument('maxRadius', type=int, default=190, nargs='?')

    args = parser.parse_args()

    tespit_ve_ciz(args)

