import pdf2image as p2i

images = p2i.convert_from_path('Bosworth.pdf')
cnt = 0
for image in images:
    cnt += 1
    image.save("outputs/Bosworth"+str(cnt)+".jpg", "JPEG")

