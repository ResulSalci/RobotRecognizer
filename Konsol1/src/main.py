import sys
import cv2
import numpy as np
import os


if __name__ == "__main__":
    if len(sys.argv) == 7 or len(sys.argv) == 1:
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

            if len(sys.argv) == 7 :
                lower_hue, upper_hue, lower_saturation, upper_saturation, lower_value, upper_value = map(int, sys.argv[1:])
                lower_color = np.array([lower_hue, lower_saturation, lower_value])
                upper_color = np.array([upper_hue, upper_saturation, upper_value])
            if len(sys.argv) == 1:
                #Beyaz renk için alt üst sınır belirliyorum.
                lower_color = np.array([0, 0, 200])
                upper_color = np.array([255, 30, 255])

            #Color Thresholding
            color_mask = cv2.inRange(hsv_image, lower_color, upper_color)

            #Contour belirliyorum
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #Contour alanını filitreliyorum
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

            if len(filtered_contours) > 0:
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
                cv2.waitKey(0)  # Resmi göster ve bir tuşa basmayı bekle
                cv2.destroyAllWindows()  # Tüm pencereleri kapat
            else:
                print(f"No robots detected in the image: {image_file}")

        sys.exit(0)
else:
    print("Usage is: python3 main.py <lower_hue> <upper_hue> <lower_saturation> <upper_saturation> <lower_value> <upper_value>\nUsage is : python3 main.py")
    sys.exit(1)