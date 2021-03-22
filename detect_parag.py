import cv2
import numpy as np

image = cv2.imread('outputs/Bosworth20.jpg')
# grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Gaussian blur
blur = cv2.GaussianBlur(gray, (5,5), 0)
# Otsu's threshold
# thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,23,3)


# Create rectangular structuring element and dilate
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.dilate(thresh, kernel, iterations=4)

# Find contours and draw rectangle
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = cnts[0] if len(cnts) == 2 else cnts[1]
crop_cnt = 0
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)
    area = cv2.contourArea(c)
    if area < 10:
        cv2.drawContours(image, [c], -1, (36, 255, 12), 1)
    # crop the rectangles and save
    crop_img = image[y:y + h, x:x + w]
    # cv2.imwrite("outputs/Bosworth_cropped" + str(crop_cnt) + ".png", crop_img)
    cv2.imshow("outputs/Bosworth_cropped", crop_img)
    cv2.waitKey(0)
    crop_cnt = crop_cnt + 1


cv2.imshow('thresh', thresh)
cv2.waitKey()
cv2.imshow('dilate', dilate)
cv2.waitKey()
cv2.imshow('image', image)
# cv2.imwrite('/outputs/explain-nested-dissection5.jpg', image)
cv2.waitKey()