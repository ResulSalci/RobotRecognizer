import cv2
import numpy as np
import os

image_folder = '/Users/muhammedemincaglar/Documents/GitHub/RobotRecognizer/Konsol1/input'
output_folder = '/Users/muhammedemincaglar/Documents/GitHub/RobotRecognizer/Konsol1/output'

# Klasördeki tüm dosyaları al
image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

# Her bir görüntü üzerinde işlem yap
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Failed to load image: {image_path}")
        continue
    height, width, _ = image.shape
    # (width, height)
    if height == 4032:
        image = cv2.resize(image, (600, 802))
    else:
        image = cv2.resize(image, (802, 600))
    #RGB'den HSV'ye dönüştürdüm
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    #Beyaz renk için alt üst sınır belirliyorum.
    lower_color = np.array([0, 0, 200])
    upper_color = np.array([255, 30, 255])

    #Color Thresholding
    color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

    #Contour belirliyorum
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Contour alanını filitreliyorum
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    #Her bir robotu işaretle
    for i, contour in enumerate(filtered_contours):
        #Color thresholding yapılan görüntüdeki çıktı üstünde robotların konumunu alıyorum
        M = cv2.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = cv2.contourArea(contour)

        #Robotun çevresini belirliyorum.
        cv2.drawContours(image, [contour], -1, (0, 0, 255), 2)

        #Robotun merkezine belirlenen robotun etiket atıyorum
        cv2.putText(image, f"Robot {i+1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow(f'Detected Robots - {image_file}', image)
    result_image_path = os.path.join(output_folder, f'sonuc_{image_file}')
    cv2.imwrite(result_image_path, image)

cv2.waitKey(0)
cv2.destroyAllWindows() 