from PIL import Image
import matplotlib.pyplot as plt
import argparse
import logging
import os
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import SimpleITK as sitk
from utils.dataset import BasicDataset
from BiSeNet.networkAttn import BiSeNet
from SegGradTest import Guided_backprop,normalize
from utils.data_vis import plot_img_and_mask
import os.path
import glob
import PIL.Image as Image
import numpy as np
from os.path import splitext
from torchvision import transforms
from PIL import Image

import matplotlib.pyplot as plt
from seggradcam import SegGradCAM,SuperRoI,BiasRoI,ClassRoI,PixelRoI
from visualize_sgc import SegGradCAMplot
import pylab
# 加载网络模型
pretrained_model = "D:/TeaNet/BiSeNet_2/BiSeNet/source/pytorch-model/resnet18_v1.pth"
criterion = nn.CrossEntropyLoss()
# net = UNet(n_channels=3, n_classes=4)
net = BiSeNet(out_planes=4, is_training=True,
              criterion=criterion,
              pretrained_model=None,
              )
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f'Loading model MODEL.pth')
logging.info(f'Using device {device}')

net.to(device=device)
net.load_state_dict(torch.load('./MODEL.pth', map_location=device))


def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)

def Calculatedice(input,target):
    eps=1e-7
    inter = torch.dot(input.view(-1), target.view(-1))
    union = torch.sum(input) + torch.sum(target) + eps
    dice = (2 * inter.float() + eps) / union.float()
    return dice

def CalculateTensorDice(inputTensor,targetDir):
    # 处理mask文件，接着逐channel计算dice值
    print("targetDir",targetDir)
    mask=Image.open(targetDir)
    mask=torch.from_numpy(BasicDataset.preprocess(mask,1.0))

    mask=mask.squeeze().type(torch.int64)
    print('mask', mask.shape,type(mask),mask.dtype)
    mask=get_one_hot(mask,4)
    mask=mask.permute(2,0,1)
    list=[]
    print(inputTensor.dtype,mask.dtype)
    for i in range(0,inputTensor.shape[0]):

        dice_i=Calculatedice(inputTensor[i],mask[i])
        list.append(dice_i)
    print("dice0:",list[0],"dice1:",list[1],"dice2:",list[2],"dice3:",list[3])
    file_handle = open('DiceResult.txt', mode='a')
    file_handle.write(splitext(os.path.basename(targetDir))[0])
    file_handle.write("\n")
    newStr="dice0:"+str(list[0])+"dice1:"+str(list[1])+"dice2:"+str(list[2])+"dice3:"+str(list[3])+"\n"
    file_handle.write(newStr)
    file_handle.close()





def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


def predictMask(jpgfile,outdirMHD,maskDir,outdir,net,device):
    img=Image.open(jpgfile)
    # try:
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(img,1.0))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    img_height=img.size()[2]
    img_width=img.size()[3]

    print(img.size()[2],img.size()[3])
    with torch.no_grad():
        output = net(img)
        print(output.shape,output.size())
        # 订正顺序
        # output = output.squeeze(0)
        # index = [0, 3,1,2]
        # output = output[index]
        # output = output.unsqueeze(0)

        output_height=output.size()[2]
        output_width=output.size()[3]
        print('output',output_width,output_height,output.size()[1])
        diffY=img_height-output_height
        diffX=img_width-output_width
        output=F.pad(output,[diffX // 2, diffX - diffX // 2,
                                              diffY // 2, diffY - diffY // 2])

        # 将分割结果可视化处理
        probs = F.softmax(output, dim=1)[0]
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((778,549)),
            transforms.ToTensor()
        ])

        full_mask = tf(probs.cpu()).squeeze()
        output=output.squeeze()

        softmax_func = nn.Softmax(dim=0)
        output= softmax_func(output)
        # output = tf(output.cpu()).squeeze()
        output=(output>0.5).float()
        output=output.cpu()
        print(output.shape)
        targetBaseDir=splitext(jpgfile)[0]+'_gt.png'

        # print("patientName",strName)
        targetDir=os.path.join(maskDir,os.path.basename(targetBaseDir))

        print("保存为mhd文件之前")
        print("output的shape",output.shape,type(output),output.dtype)
        CalculateTensorDice(output,targetDir)
        # 保存为mhd文件之前逐channel计算一次dice值
        pred_output=output.argmax(0).cpu()
        print('pred_output',pred_output.shape)
        list=[]
        # for i in range(778):
        #     for j in range(549):
        #         x=pred_output[i][j]
        #         if x not in list:
        #             list.append(x)
        # print('list',list)


        mhd_data = sitk.GetImageFromArray(pred_output)
        baseName = splitext(jpgfile)[0]+'.mhd'
        strName = os.path.basename(jpgfile)[0:11]
        makeDirMHD = outdirMHD + strName + "/"
        if not os.path.exists(makeDirMHD):
            os.makedirs(makeDirMHD)
        save_dir = os.path.join(makeDirMHD,os.path.basename(baseName))
        sitk.WriteImage(mhd_data, save_dir)
        #     mhd文件保存完毕，
        #     读取保存的mhd文件，

        data = sitk.ReadImage(save_dir)
        scan = sitk.GetArrayFromImage(data)

        picture = scan.squeeze()

        # 保存为mhd文件后再计算一次dice值，比对俩次dice值是否相同
        # predImg=Image.open(picture)
        # predImg= torch.from_numpy(BasicDataset.preprocess(predImg, 1.0))
        print("保存为mhd文件之后")

        pictureDice=torch.tensor(picture).to(torch.int64)
        pictureDice = get_one_hot(pictureDice, 4)
        pictureDice = pictureDice.permute(2, 0, 1)
        print("pictureDice", pictureDice.shape, type(pictureDice), pictureDice.dtype)
        CalculateTensorDice(pictureDice,targetDir)
        plt.imshow(picture)
        pylab.show()
        print(picture.shape)








    pred_mask= F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
    result=mask_to_image(pred_mask)
    result.save(os.path.join(outdir,os.path.basename(jpgfile)))
    strName = os.path.basename(jpgfile)[0:11]
    makeDir = outdir + strName + "/"
    if not os.path.exists(makeDir):
        os.makedirs(makeDir)
    targetSaveDir = os.path.join(makeDir, os.path.basename(jpgfile))
    result.save(targetSaveDir)
    # 可视化某一张图片
    I = Image.open(jpgfile).convert('RGB')
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size = 224

    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(means, stds)
    ])

    # tensor = transform(I).unsqueeze(0).requires_grad_()
    # tensor = tensor.to(device=device, dtype=torch.float32)
    #
    # guided_bp = Guided_backprop(net)
    #
    # result = guided_bp.visualize(tensor, 3)
    #
    # result = normalize(result)
    # plt.imshow(result[52])
    # plt.show()

    # segCAM
    # set up layers for propogation
    prop_from_layer=''
    # prop_to_layer=''
    clsroi=ClassRoI(cls=3,pred_result=output)
    pixsgc=SegGradCAM(net,img,cls=3,prop_to_layer='context_path.layer0.1.bn2',roi=clsroi)
    pixsgc.SGC()
    groundTruth=Image.open(targetDir)

    plotter=SegGradCAMplot(pixsgc,model=net,n_classes=4,outfolder="D:/Tea3/BiSeNet_3/SegGradCAM/",gt=groundTruth)
    plotter.explainBase('picture1','picture2','picture3')






        # new_img.save(os.path.join(outdir,os.path.basename(jpgfile)))
    # except Exception as e:
    #     print(e)




for jpgfile in glob.glob("D:/InstrumentProject/CAMUS/training/pictures_8/*.jpg"):
    print(jpgfile)
    predictMask(jpgfile,outdirMHD="D:/Tea3/BiSeNet_3/SaveMHD/",outdir="D:/Tea3/BiSeNet_3/predMasks/",maskDir="D:/InstrumentProject/CAMUS/training/masks_6/",net=net,device=device)