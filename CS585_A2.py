import cv2
import numpy as np
import math

def find_defects(binary_frame, crop_img):
    #find the contours from the thresholded blurred image
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #get the largest contour (shape in the image)
    largest_contour = max(contours, key=lambda x: cv2.contourArea(x))
    #get the coordinates of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    #draw a rectangle the original cropped image
    cv2.rectangle(crop_img, (x, y), (x+w, y+h), (0, 0, 255), 0)
    #the convexHull function returns the smallest convex polygon that encloses the contour
    hull = cv2.convexHull(largest_contour)
    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [largest_contour], 0, (0, 255, 0), 0)
    cv2.drawContours(drawing, [hull], 0, (0, 255, 0), 0)
    hull = cv2.convexHull(largest_contour, returnPoints=False)
    #finds the defects in the contour (helps in identifying the fingers)
    defects = cv2.convexityDefects(largest_contour, hull)
    return defects, contours, largest_contour, drawing


def vid_capture():
    vid = cv2.VideoCapture(0)

    while True:
        ret, img = vid.read()
        if not ret:
            break  # Break the loop if no frame is read

        # Perform operations on each frame
        cv2.rectangle(img, (250, 250), (50, 50), (0, 255, 0), 0)
        #crop the frame image
        crop_img = img[50:250, 50:250]
        #convert the frame to grayscale
        gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        value = (35, 35)
        #blurs the grayscaled cropped image
        blurred = cv2.GaussianBlur(gray_img, value, 0)
  
        #apply thresholding on the blurred grayscale image
        _, binary_frame = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        #cv2.imshow('Blurred', binary_frame)

        defects, contours, largest_contour, drawing = find_defects(binary_frame, crop_img)

        count_defects = 0
        cv2.drawContours(binary_frame, contours, -1, (0, 255, 0), 3)
        for i in range(defects.shape[0]):
            df = defects[i, 0]
            start = tuple(largest_contour[df[0]][0])
            end = tuple(largest_contour[df[1]][0])
            far = tuple(largest_contour[df[2]][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1]- start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1])**2)
            angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) 

            if angle <= np.pi/2:
                count_defects += 1
                cv2.circle(crop_img, far, 1, [0, 0, 255], -1)
            cv2.line(crop_img, start, end, [0, 255, 0], 2)

        if count_defects == 0:
            cv2.putText(img, "1 finger", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        elif count_defects == 1:
            cv2.putText(img, "2 fingers", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        elif count_defects == 2:
            cv2.putText(img, "3 fingers", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        elif count_defects == 3:
            cv2.putText(img, "4 fingers", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        elif count_defects == 4:
            cv2.putText(img, "5 fingers", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow('Image', img)
        all_img = np.hstack((drawing, crop_img))
        cv2.imshow('Contours', all_img)
        k = cv2.waitKey(10)
        if k == 27:
            break
    vid.release()
    cv2.destroyAllWindows()

def main():
    vid_capture()



if __name__ == "__main__":
    main()
