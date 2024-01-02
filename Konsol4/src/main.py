import cv2
import os
import argparse

def contour_detection(blurparam1, blurparam2, blurparam3, cannyparam1, cannyparam2):
    # Görüntülerin bulunduğu klasör yolu
    input_folder = "../input"

    # İşlenmiş görüntülerin yazılacağı klasör yolu
    output_folder = "../output"

    # input_folder içindeki görüntülere tek tek erişme
    for image_file in os.listdir(input_folder):

        # Görüntüyü yükle
        image = cv2.imread(os.path.join(input_folder, image_file))

        # Gri tonlamaya dönüştür
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Gürültüyü azaltmak için Gaussian Blur uygula
        blur = cv2.GaussianBlur(gray, (blurparam1, blurparam2), blurparam3)

        # Kenarları belirleme (Canny Edge Detection)
        edges = cv2.Canny(blur, cannyparam1, cannyparam2)

        # Contour (sınırlar) tespiti
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Konturları görüntünün üzerine çiz
        cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

        # İşlenmiş görüntüyü yazdırma
        cv2.imwrite(os.path.join(output_folder, image_file), image)

if __name__ == '__main__':
    # Konsoldan önemli parametrelerin alınması (bir değer girilmezse default değerler alınır.)
    parser = argparse.ArgumentParser(description='Değişkenlik gösteren parametrelerin konsoldan alınması.')

    parser.add_argument('blurparam1', type=int, default=35, nargs='?')
    parser.add_argument('blurparam2', type=int, default=35, nargs='?')
    parser.add_argument('blurparam3', type=int, default=2, nargs='?')
    parser.add_argument('cannyparam1', type=int, default=25, nargs='?')
    parser.add_argument('cannyparam2', type=int, default=200, nargs='?')

    args = parser.parse_args()

    contour_detection(args.blurparam1, args.blurparam2, args.blurparam3, args.cannyparam1, args.cannyparam2)



