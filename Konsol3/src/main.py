import cv2
import os

def detect_contours(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (19, 19), 0)
    edges = cv2.Canny(blurred, 40, 130)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def main():
    input_folder = "../input"
    output_folder = "../output"
    for image_file in os.listdir(input_folder):
        image = cv2.imread(os.path.join(input_folder, image_file))
        detected_contours = detect_contours(image)
        for contour in detected_contours:
            cv2.drawContours(image, [contour], -1, (0,255, 0), 7)
        cv2.imwrite(os.path.join(output_folder, image_file), image)

if __name__ == "__main__":
    main()
