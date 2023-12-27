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
        return np.array([[24,14,56], [87,91,175]])
    elif(number >= 3 and number <= 4):
        return np.array([[59,28,114], [89,94,163]])
    elif(number == 5):
        return np.array([[30,25,45], [80,97,138]])
    elif(number == 6):
        return np.array([[27,30,25], [80,160,142]])
    elif(number >= 7 and number <= 8):
        return np.array([[30,28,33], [97,155,160]])
    elif(number >= 9 and number <= 10):
        return np.array([[67,48,58], [97,102,255]])
    elif(number >= 11 and number <= 12):
        return np.array([[41,39,45], [97,87,255]])
    elif(number >= 13 and number <= 14):
        return np.array([[38,53,117], [56,101,255]])
    elif(number >= 15 and number <= 17):
        return np.array([[0,31,45], [64,95,255]])
    elif(number >= 18 and number <= 22):
        return np.array([[0,40,111], [66,85,255]])
    elif(number >= 23 and number <= 24):
        return np.array([[31,20,106], [78,69,255]])
    elif(number >= 25 and number <= 27):
        return np.array([[33,11,116], [80,58,255]])
    elif(number >= 28 and number <= 49):
        return np.array([[75,25,165], [100,55,255]])
    elif(number >= 50 and number <= 62):
        return np.array([[47,25,83], [100,90,255]])
    elif(number >= 63 and number <= 69):
        return np.array([[42,5,83], [107,75,170]])
    elif(number >= 70 and number <= 79):
        return np.array([[50,0,160], [179,53,255]])
    elif(number >= 80 and number <= 89):
        return np.array([[20,0,11], [132,130,74]])
    elif(number >= 90 and number <= 94):
        return np.array([[65,0,45], [131,124,171]])
    elif(number >= 95 and number <= 99):
        return np.array([[100,0,53], [141,120,181]])
    elif(number >= 100 and number <= 108):
        return np.array([[100,20,63], [122,125,153]])
    else:
        return None

image_folder = '/Users/muhammedemincaglar/Documents/GitHub/RobotRecognizer/Konsol2/input'
output_folder = '/Users/muhammedemincaglar/Documents/GitHub/RobotRecognizer/Konsol2/output'

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

    color_range = get_color_from_file(file_num)

    #Color Thresholding
    color_mask = cv2.inRange(hsv_image, color_range[0], color_range[1])

    #Contour belirliyorum
    contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Contour alanını filitreliyorum
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

    #Her bir robotu işaretle
    if len(filtered_contours) > 0:
        # En büyük alana sahip konturu bul
        largest_contour = max(filtered_contours, key=cv2.contourArea)

        # Robotu işaretleme ve etiketleme işlemleri
        M = cv2.moments(largest_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        area = cv2.contourArea(largest_contour)

        cv2.drawContours(image, [largest_contour], -1, (0, 0, 255), 2)
        cv2.putText(image, "Robot", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        print("Hiç robot bulunamadı.")

    cv2.imshow(f'Detected Robots - {image_file}', image)
    result_image_path = os.path.join(output_folder, f'sonuc_{image_file}')
    cv2.imwrite(result_image_path, image)

cv2.waitKey(0)
cv2.destroyAllWindows()



