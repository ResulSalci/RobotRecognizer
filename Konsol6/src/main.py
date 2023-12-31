import cv2
import numpy as np
import sys

def find_robot(image_org, minArea, maxArea, minCircularity, gaussianBlurKernelSize, thresholdMin, thresholdMax):
    
    #Görselin yüklekliği ve genişliğini al 
    height = image_org.shape[0]
    width = image_org.shape[1]

    #(width, height)
    # Yükseklik 4032 pixel is görsel dikeydir değilse yataydır. Yeniden boyutlandırmada dikkate alınır.
    if height == 4032:
        image_org = cv2.resize(image_org, (600, 802))

    else:
        image_org = cv2.resize(image_org, (802, 600))

    # Görsele girilen kernel boyuttunu kullanarak bir gaussian blur uygula
    image = cv2.GaussianBlur(image_org, (gaussianBlurKernelSize, gaussianBlurKernelSize), 0)
    
    #192, 193
    # Görseli blob deetecion'ın anlayabileceği bir hale getirmek için girilen değerleri kullanarak bir thresholding yaP.
    image = cv2.threshold(image, thresholdMin, thresholdMax, cv2.THRESH_BINARY_INV)[1]

    # Blob detector'a verilecek paramtereler objesini oluştur
    params = cv2.SimpleBlobDetector_Params()

    # Alana göre blob bulmayı aktif et
    params.filterByArea = True

    # Blob alanının alabileceği minimum değer
    params.minArea = minArea

    # Blob alanının alabileceği maksimum değer
    params.maxArea = maxArea

    # Blobların şekillerinin daireye yakınlığına göre ayırmayı aktif et
    params.filterByCircularity = True

    # Bir blobun en az ne kadar daire olması gerektiği şartını ver
    params.minCircularity = minCircularity

    # parametre objesini kullanarak bir blob detector oluştur
    detector = cv2.SimpleBlobDetector_create(params)

    # detectoru kullnarak blobları tespit et
    keypoints = detector.detect(image)

    # Orijinal resimde blobların olduğu yerler merkez olacak şekilde kırmızı daireler çiz
    image_keypoints = np.zeros_like(image_org)
    image_keypoints = cv2.drawKeypoints(image_org, keypoints, image_keypoints, (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    return image_keypoints

def analyse_images(minArea, maxArea, minCircularity, gaussianBlurKernelSize, thresholdMin, thresholdMax):
    for i in range(32):
        # Resmi diskten oku
        image_org = cv2.imread(f'Konsol6/input/{i+1}.jpg', cv2.IMREAD_GRAYSCALE)

        # Resimi analiz et ve blobları bul
        image_keypoints = find_robot(image_org, minArea, maxArea, minCircularity, gaussianBlurKernelSize, thresholdMin, thresholdMax)

        # Bloblar işaretlenmiş olan resmi output klasörüne yaz
        cv2.imwrite(f"Konsol6/output/{i+1}.jpg", image_keypoints)

if __name__ == "__main__":

    # minArea = 300, maxArea = 1125000, minCircularity = 0.40, gaussianBlurKernelSize = 51, thresholdMin = 192, thresholdMax = 193
    if len(sys.argv) != 7:
        print("Usage is: python main.py <minArea> <maxArea> <minCircularity> <gaussianBlurKernelSize> <thresholdMin> <thresholdMax>")
        sys.exit(1)

    analyse_images(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))