from PIL import Image
import matplotlib.pyplot as plt


im = Image.open('grayYUV.png')
pixelMap = im.load()

img = Image.new( im.mode, im.size)
pixelsNew = im.load()
for i in range(img.size[0]):
    for j in range(img.size[1]):
        if 205 in pixelMap[i,j]:
           pixelMap[i,j] = (0,0,0,255)
        pixelsNew[i,j] = pixelMap[i,j]
img.show()       
im.save("new.png")
img.close()
