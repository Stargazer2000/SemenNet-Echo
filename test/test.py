import os
import sys
import time
import argparse
from collections import OrderedDict, defaultdict
import sys
sys.path.append("./BiSeNet")
from BiSeNet.resnet import resnet18
from BiSeNet.resnet import ResNet,BasicBlock
import torch
import torch.utils.model_zoo as model_zoo
import torch.distributed as dist
import torch.nn as nn
from utils.dataset import BasicDataset
from PIL import Image


import logging
model_file='D:/Tea/BiSeNet/BiSeNet/source/pytorch-model/resnet18_v1.pth'
if isinstance(model_file, str):
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    print(state_dict.keys())
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k
        new_state_dict[name] = v
    # print('new_state_dict:',new_state_dict.keys())
model=ResNet(BasicBlock, [2, 2, 2, 2])
model.load_state_dict(state_dict, strict=False)
# print('ResNet module的关键字',model.state_dict().keys())
print(model.state_dict().keys())
ckpt_keys = set(state_dict.keys())
own_keys = set(model.state_dict().keys())
missing_keys = own_keys - ckpt_keys
print('missing keys',missing_keys)
unexpected_keys = ckpt_keys - own_keys
print('unexpected_keys:',unexpected_keys)

if len(missing_keys) > 0:
    logging.info('Missing key(s) in state_dict: {}'.format(
        ', '.join('{}'.format(k) for k in missing_keys)))

if len(unexpected_keys) > 0:
    logging.info('Unexpected key(s) in state_dict: {}'.format(
        ', '.join('{}'.format(k) for k in unexpected_keys)))




def load_model(model, model_file, is_restore=False):
    t_start = time.time()
    if isinstance(model_file, str):
        state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in state_dict.keys():
            state_dict = state_dict['model']
    else:
        state_dict = model_file
    t_ioend = time.time()
    print("t_ioend: ",time.time())

    if is_restore:
        new_state_dict = OrderedDict()
        print('new_state_dict:',new_state_dict.keys(),new_state_dict.values())
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    ckpt_keys = set(state_dict.keys())
    own_keys = set(model.state_dict().keys())
    missing_keys = own_keys - ckpt_keys
    unexpected_keys = ckpt_keys - own_keys

    if len(missing_keys) > 0:
        logging.info('Missing key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in missing_keys)))

    if len(unexpected_keys) > 0:
        logging.info('Unexpected key(s) in state_dict: {}'.format(
            ', '.join('{}'.format(k) for k in unexpected_keys)))

    del state_dict
    t_end = time.time()
    logging.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model

class BiSeNet(nn.Module):

    def __init__(self, out_planes,
                 pretrained_model=None,
                 norm_layer=nn.BatchNorm2d):
        bn_eps = 1e-5
        bn_momentum=0.1
        self.n_classes=out_planes

        self.out_planes=out_planes
        super(BiSeNet, self).__init__()
        self.context_path = resnet18(pretrained_model, norm_layer=norm_layer,
                                     bn_eps=bn_eps,
                                     bn_momentum=bn_momentum,
                                     deep_stem=False, stem_width=64)
    def forward(self, data, label=None):
        spatial_out = self.spatial_path(data)

        context_blocks = self.context_path(data)
        print(type(context_blocks))
        print("context_path:",context_blocks)
        return context_blocks




pretrained_model = 'D:/Tea/BiSeNet/BiSeNet/source/pytorch-model/resnet18_v1.pth'
full_img="D:/Tea/CAMUS/training/pictures_3/patient0001_2CH_ED.jpg"
full_img=Image.open(full_img)
print(type(full_img))
scale_factor=0.5
img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
img = img.unsqueeze(0)

context_blocks = BiSeNet(img)
print(context_blocks)



