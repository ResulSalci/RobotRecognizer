import cv2
import numpy as np
import argparse

def get_color_from_file(number):
    if(number >= 1 and number <= 2):
        return np.array([[24,16,56], [87,91,171]])
    elif(number >= 3 and number <= 4):
        return np.array([[59,30,95], [89,94,163]])
    elif(number >= 5 and number <= 8):
        return np.array([[30,30,34], [89,142,166]])
    elif(number >= 9 and number <= 12):
        return np.array([[54,47,50], [98,94,255]])
    elif(number >= 13 and number <= 14):
        return np.array([[38,53,117], [56,100,255]])
    elif(number >= 15 and number <= 22):
        return np.array([[0,36,81], [66,110,255]])
    elif(number >= 23 and number <= 27):
        return np.array([[30,16,113], [80,64,255]])
    elif(number >= 28 and number <= 49):
        return np.array([[78,17,169], [99,55,255]])
    elif(number >= 50 and number <= 69):
        return np.array([[44,16,86], [105,78,212]])
    elif(number >= 70 and number <= 79):
        return np.array([[64,3,163], [150,50,255]])
    elif(number >= 80 and number <= 89):
        return np.array([[20,0,11], [139,131,75]])
    elif(number >= 90 and number <= 94):
        return np.array([[65,0,42], [131,150,190]])
    elif(number >= 95 and number <= 108):
        return np.array([[100,11,56], [132,122,166]])
    else:
        return None

def find_robot(image_org, i, minArea, maxArea, gaussianBlurKernelSize):

    # Görseli küçült
    image_org = cv2.resize(image_org, (600, 451))

    # Görseli BGR'dan HSV renk uzayına geçir
    image = cv2.cvtColor(image_org, cv2.COLOR_BGR2HSV)

    # Girilen parametereleri kullnarak görsele bir gaussian blur uygula
    image = cv2.GaussianBlur(image, (gaussianBlurKernelSize, gaussianBlurKernelSize), 0)

    # Uygun renk aralıklarını getir
    color_range = get_color_from_file(i+1)

    # Renk aralıklarını kullnarak maskeleme yap
    image = cv2.inRange(image, color_range[0], color_range[1])
    
    # Blob siyah renge göre aradığı ve robotun büyük bir kısmı beyaz olduğu için resmin renklerini ters çevir 
    image = cv2.bitwise_not(image)

    # Blob detector'a verilecek paramtereler objesini oluştur
    params = cv2.SimpleBlobDetector_Params()

    # Alana göre blob bulmayı aktif et
    params.filterByArea = True

    # Blob alanının alabileceği minimum değer
    params.minArea = minArea

    # Blob alanının alabileceği maksimum değer
    params.maxArea = maxArea

    # parametre objesini kullanarak bir blob detector oluştur
    detector = cv2.SimpleBlobDetector_create(params)


    # detectoru kullnarak blobları tespit et
    keypoints = detector.detect(image)

    # Orijinal resimde blobların olduğu yerler merkez olacak şekilde kırmızı daireler çiz
    image_keypoints = np.zeros_like(image_org)
    image_keypoints = cv2.drawKeypoints(image_org, keypoints, image_keypoints, (0, 0, 255),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return image_keypoints

#108
def analyse_images(minArea, maxArea, gaussianBlurKernelSize):
    for i in range(108):

        if i == 15:
            continue

        image_org = cv2.imread(f'Konsol7/input/{i+1}.jpg')
        
        image_keypoints = find_robot(image_org,i, minArea, maxArea, gaussianBlurKernelSize)

        cv2.imwrite(f"Konsol7/output/{i + 1}.jpg", image_keypoints)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Analiz Parametleri")

    parser.add_argument("minArea", type=int, default = 1500, nargs="?")
    parser.add_argument("maxArea", type=int, default = 1125000, nargs="?")
    parser.add_argument("gaussianBlurKernelSize", type=int, default = 51, nargs="?")

    args = parser.parse_args()

    analyse_images(args.minArea, args.maxArea, args.gaussianBlurKernelSize)