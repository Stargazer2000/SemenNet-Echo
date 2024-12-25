import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset
from dice_loss import dice_coeff,multi_dice_coeff,multi_dice_coeff_list
list2=[]
scale_factor=1.0
# full_img=Image.open("D:/InstrumentProject/CAMUS/training/masks_6/patient0013_4CH_ES_gt.png")
full_img=Image.open("./img1.png")
mask=Image.open("./output1002.jpg")
# full_img=Image.open("D:/Tea/CAMUS/training/masks_6/patient0003_2CH_ED_gt.png")
for i in range(800):
    for j in range(800):
        x=full_img.getpixel((i,j))
        if x not in list2:
            list2.append(x)
        # if i > threshold1 and i < threshold2:
        # if x == 3:
        #     full_img.putpixel((i,j),179)
        #
        # else:
        #     full_img.putpixel((i, j), 33)


def Dice(input, target):

    eps = 0.0001
    print("这是进行view变换后的shape:")
    print((input.view(-1)).shape)
    print((target.view(-1)).shape)
    print(input.shape,target.shape)
    if input.shape != target.shape:
        raise ValueError("Shape mismatch: input and target must have the same shape.")
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target) + eps

    t = (2 * inter.float() ) / union.float()
    # inter = (input * target).sum(axis=[-1, -2])
    # union = (input + target).sum(axis=[-1, -2])
    # dice = (2. * inter / (union + eps)).mean(axis=-1)
    # return dice
    return t






list2.sort()
print('list2:',list2)


mask=torch.from_numpy(BasicDataset.preprocess(mask, scale_factor))
mask=torch.squeeze(mask)
img=torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
img=torch.squeeze(img)
print('type:',type(img),type(mask))
img=img.to(dtype=torch.float32)
mask=mask.to(dtype=torch.float32)
dice=Dice(img,mask)
print(mask.shape)
print(img.shape)
print('dice:',dice)
