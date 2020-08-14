import pdf2image as p2i

images = p2i.convert_from_path('paper-final-2014Mar.pdf')
cnt = 0
for image in images:
    cnt += 1
    image.save("output/test"+str(cnt)+".jpg", "JPEG")

