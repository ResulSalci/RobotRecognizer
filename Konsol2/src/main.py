import sys
import cv2
import numpy as np
import os

def get_number_from_filename(filename):
    file_number = int (filename.split('.')[0])
    if(file_number <= 0 and file_number >= 109):
        return None
    else:
        return int(file_number)
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

if __name__ == "__main__":
    if len(sys.argv) == 7 or len(sys.argv) == 1:
        image_folder = '../input'
        output_folder = '../output'

        # Klasördeki tüm dosyaları al
        image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

        # Her bir görüntü üzerinde işlem yap
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path)
            height, width, _ = image.shape
            
            # (width, height)
            if height == 4032:
                image = cv2.resize(image, (600, 802))
            else:
                image = cv2.resize(image, (802, 600))
            
            file_num = get_number_from_filename(image_file)
            
            #RGB'den HSV'ye dönüştürdüm
            hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            color_range = None
            if len(sys.argv) == 1:
                color_range = get_color_from_file(file_num)
            if len(sys.argv) == 7:
                lower_hue, upper_hue, lower_saturation, upper_saturation, lower_value, upper_value = map(int, sys.argv[1:])
                color_range = [np.array([lower_hue, lower_saturation, lower_value]),
                           np.array([upper_hue, upper_saturation, upper_value])]
            #Color Thresholding
            color_mask = cv2.inRange(hsv_image, color_range[0], color_range[1])

            #Contour belirliyorum
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            #Contour alanını filitreliyorum
            filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

            #Her bir robotu işaretle
            if len(filtered_contours) > 0:
                for cnt in filtered_contours:  # Her konturu dolaş
                    M = cv2.moments(cnt)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    area = cv2.contourArea(cnt)

                    cv2.drawContours(image, [cnt], -1, (0, 0, 255), 2)
                    cv2.putText(image, "Robot", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                print("Hiç robot bulunamadı.")

            cv2.imshow(f'Detected Robots - {image_file}', image)
            result_image_path = os.path.join(output_folder, f'sonuc_{image_file}')
            cv2.imwrite(result_image_path, image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

else:
    print("Usage is: python3 main.py <lower_hue> <upper_hue> <lower_saturation> <upper_saturation> <lower_value> <upper_value>\nUsage is : python3 main.py")
    sys.exit(1)

