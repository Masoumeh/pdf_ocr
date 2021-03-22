# import the necessary packages
import imutils
import cv2
import numpy as np

# Load image, convert to HSV format, define lower/upper ranges, and perform
# color segmentation to create a binary mask
image = cv2.imread('outputs/Bosworth3.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
lower = np.array([0, 0, 218])
upper = np.array([157, 54, 255])
mask = cv2.inRange(hsv, lower, upper)

# Create horizontal kernel and dilate to connect text characters
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
dilate = cv2.dilate(mask, kernel, iterations=5)

# Find contours and filter using aspect ratio
# Remove non-text contours by filling in the contour
cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    ar = w / float(h)
    if ar < 5:
        cv2.drawContours(dilate, [c], -1, (0,0,0), -1)


# Bitwise dilated image with mask, invert, then OCR
result = 255 - cv2.bitwise_and(dilate, mask)
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.imshow('dilate', dilate)
cv2.waitKey(0)
cv2.imshow("img", result)
cv2.waitKey(0)
# data = pytesseract.image_to_string(result, lang='eng',config='--psm 6')
# print(data)