import cv2 as cv
import glob
from os.path import splitext
import os
gray = cv.imread('./img_1.png.jpg')
# cv.imshow('gray', gray)
import numpy as np
import PIL.Image as Image


def ChangeColor(jpgfile,targetfile):
    gray = cv.imread(jpgfile)
    g_autumn = cv.applyColorMap(gray, cv.COLORMAP_INFERNO)
    cv.imshow('g_autumn', g_autumn)
    baseDir=splitext(jpgfile)[0]+'.jpg'

    cv.imwrite(os.path.join(targetfile, os.path.basename(jpgfile)), g_autumn)






#
# for jpgfile in glob.glob("D:/Tea3/BiSeNet_3/GoodPictures/MyOwnModule/*.jpg"):
#     print(jpgfile)
#     ChangeColor(jpgfile,targetfile="D:/Tea3/BiSeNet_3/GoodPictures/MyOwnModuleChangeColor/")

gray = cv.imread("./img.png")

g_autumn = cv.applyColorMap(gray, cv.COLORMAP_INFERNO)
cv.imshow('g_autumn', g_autumn)
# cv.waitKey()
cv.imwrite('./result3.jpg', g_autumn)



# list=[]
# # img=Image.open("D:/InstrumentProject/CAMUS/training/masks_6/patient0004_4CH_ES_gt.png")
# img=Image.open("./patient0046_2CH_ES.jpg")
# print(img.size)
# for i in range(img.size[0]):
#     for j in range(img.size[1]):
#         x=img.getpixel((i,j))
#         if x not in list:
#             list.append(x)
#
# print(list)