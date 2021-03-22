import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('outputs/Bosworth13.jpg', 0)

# find lines by horizontally blurring the image and thresholding
blur = cv2.blur(image, (91,9))
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blur = cv2.GaussianBlur(gray, (5,5), 0)

# cv2.imshow("blur", blur)
# cv2.waitKey()
b_mean = np.mean(blur, axis=1)/256

# hist, bin_edges = np.histogram(b_mean, bins=100)
# threshold = bin_edges[66]
threshold = np.percentile(b_mean, 66)
t = b_mean > threshold
'''
get the image row numbers that has text (non zero)
a text line is a consecutive group of image rows that 
are above the threshold and are defined by the first and 
last row numbers
'''
tix = np.where(1-t)
tix = tix[0]
lines = []
start_ix = tix[0]
print(tix.shape)
for ix in range(1, tix.shape[0]-1):
    if tix[ix] == tix[ix-1] + 1:
        continue
    # identified gap between lines, close previous line and start a new one
    end_ix = tix[ix-1]
    lines.append([start_ix, end_ix])
    start_ix = tix[ix]
end_ix = tix[-1]
lines.append([start_ix, end_ix])

l_starts = []
for line in lines:
    center_y = int((line[0] + line[1]) / 2)
    xx = 500
    for x in range(0,500):
        col = image[line[0]:line[1], x]
        # print("x: ", x)
        # print("col: ", col)
        if len(col) > 0:
            if np.min(col) < 64:
                xx = x
                break
    l_starts.append(xx)
print(l_starts)
median_ls = np.median(l_starts)

paragraphs = []
p_start = lines[0][0]
print(lines)

for ix in range(1, len(lines)):
     if l_starts[ix] > median_ls * 2:
        print("median")
        print(l_starts[ix])
        print(lines[ix])
        p_end = lines[ix][0] - 10
        paragraphs.append([p_start, p_end])
        p_start = lines[ix][0]
        # cv2.circle(image, (p_start, p_end), 10, (255,0,0), thickness=2)
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
p_img = np.array(image)
n_cols = p_img.shape[1]
print("p_img.shape")
print(p_img.shape)
print(paragraphs)
crop_cnt = 0
for paragraph in paragraphs:
    cv2.rectangle(p_img, (5, paragraph[0]), (n_cols - 5, paragraph[1]), (128, 128, 0), 5)
    crop_img = p_img[paragraph[0]:paragraph[1], 5:n_cols - 5]
    # cv2.imwrite("Bos_cropped" + str(crop_cnt) + ".png", crop_img)
    cv2.imshow("Bos_cropped", crop_img)
    cv2.waitKey(0)
    crop_cnt = crop_cnt + 1

# cv2.imwrite('paragraphs_Bos.png', p_img)