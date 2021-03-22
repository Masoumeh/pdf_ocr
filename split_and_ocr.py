import math

import pdf2image as p2i
import libtiff

import pytesseract
import cv2
from skimage import io, img_as_float
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


if __name__ == '__main__':

    pdf_path = input("Enter the path to the pdf file: ")
    #
    images = p2i.convert_from_path(pdf_path)
    page_cnt = 0
    img_files_name = pdf_path.replace(".pdf", "")
    for image in images:
        # im = Image.open(image)
        # print("1")
        im = trim(image)
        # print(im.size)
        # x, y = im.size
        # x2, y2 = math.floor(x - 50), math.floor(y - 20)
        # im = im.resize((x2, y2), Image.ANTIALIAS)
        print("1: ", im.size)
        im = im.convert("1")
        print("2: ", im.size)

        im.save(img_files_name + "_page" + str(page_cnt) + ".png") #, dpi=[100,100], optimize=True, quality=1)
        # image.save(img_files_name + "_page" + str(page_cnt) + ".png", "PNG")
        # text = pytesseract.image_to_string(image, lang='eng')
        # file = open(img_files_name + "_page" + str(page_cnt) + ".txt", "w")
        # file.write(text)
        # file.write("\n")
        page_cnt += 1


        ## (1) Convert to gray, and threshold
        # img = cv2.imread("/Users/ali/Documents/pdf_ocr/outputs/Bosworth3.jpg")
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # th, threshed = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
        #
        # ## (2) Morph-op to remove noise
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        # morphed = cv2.morphologyEx(threshed, cv2.MORPH_CLOSE, kernel)
        #
        # ## (3) Find the max-area contour
        # cnts = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cnt = sorted(cnts, key=cv2.contourArea)[-1]
        #
        # ## (4) Crop and save it
        # x, y, w, h = cv2.boundingRect(cnt)
        # dst = img[y:y + h, x:x + w]
        # cv2.imshow("001", dst)
        # cv2.waitKey()

    ######### 2 ############
        # gray = 255 * (gray < 128).astype(np.uint8)  # To invert the text to white
        # coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        # x, y, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        # rect = img[y:y + h, x:x + w]  # Crop the image - note we do this on the original image
        # cv2.imshow("Cropped", rect)  # Show it
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # scale_percent = 30  # percent of original size
        # width = int(img.shape[1] * scale_percent / 100)
        # height = int(img.shape[0] * scale_percent / 100)
        # dim = (width, height)
        # print(rect.size)

        # resize image
        # resized = cv2.resize(rect, dim, interpolation=cv2.INTER_AREA)
#         print(resized.size)

        # cv2.imwrite("crop_resize_cv.tiff", resized)  # Save the image
        # cropped_img = Image.open(rect)
        # print(cropped_img.size)
        # # cropped_img_resized = cropped_img.resize((600,900),Image.ANTIALIAS)
        # cropped_img.save("rect_resized.jpg", optimize=True, quality=10)
        # exit()

########################
    ######## 3 #############

    # image = img_as_float(io.imread('/Users/ali/Documents/pdf_ocr/outputs/Bosworth3.jpg'))

    # Select all pixels almost equal to white
    # (almost, because there are some edge effects in jpegs
    # so the boundaries may not be exactly white)
    # white = np.array([1, 1, 1])
    # mask = np.abs(image - white).sum(axis=2) < 0.05

    # Find the bounding box of those pixels
    # coords = np.array(np.nonzero(~mask))
    # top_left = np.min(coords, axis=1)
    # bottom_right = np.max(coords, axis=1)
    #
    # out = image[top_left[0]:bottom_right[0],
    #       top_left[1]:bottom_right[1]]
    #
    # plt.imshow(out)
    # plt.savefig('Bosworth3_plt_crop.png', bbox_inches='tight')
    # plt.show()

    #########################
    ####### 4 ###############
