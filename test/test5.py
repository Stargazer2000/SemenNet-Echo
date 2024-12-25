import argparse
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import torch.nn.functional as F
import numpy as np
import torch
torch.cuda.empty_cache()
import torch.nn as nn
from torch import optim
from tqdm import tqdm


from eval import eval_net
from unet import UNet
from BiSeNet.pretrainedModelsVgg import BiSeNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from BiSeNet.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
import gc



dir_img='D:/InstrumentProject/CAMUS/training/pictures_10/'
dir_mask = 'D:/InstrumentProject/CAMUS/training/masks_7/'
dataset = BasicDataset(dir_img, dir_mask, 0.2)
n_val = int(len(dataset) * 0.5)
n_train = len(dataset) - n_val
train, val = random_split(dataset, [n_train, n_val])
train_loader = DataLoader(train, batch_size=24, shuffle=False, num_workers=1, pin_memory=True)
tbar = tqdm(train_loader)
criterion = nn.CrossEntropyLoss()
pretrained_model = "D:/TeaNet/BiSeNet_2/BiSeNet/source/pytorch-model/resnet18_v1.pth"

# net = UNet(n_channels=3, n_classes=4)
net = BiSeNet(out_planes=4, is_training=True,
              criterion=criterion,
              pretrained_model=None,
              )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device=device)
# net.load_state_dict(torch.load("./MODEL.pth", map_location=device))

with torch.no_grad():
    for img, mask, id in tbar:

        img = img.cuda()
        print('img:'+img)
        pred = net(img, True)
        print('pred:'+pred)
