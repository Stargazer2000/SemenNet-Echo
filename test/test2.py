import os
import sys
import time
import argparse
from collections import OrderedDict, defaultdict
import sys

sys.path.append("./BiSeNet")
from BiSeNet.resnet import ResNet, BasicBlock
import torch
import torch.utils.model_zoo as model_zoo
import torch.distributed as dist
from BiSeNet.resnet import resnet18
from BiSeNet.resnet import ResNet, BasicBlock
from utils.dataset import BasicDataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
from BiSeNet.seg_oprs import ConvBnRelu, AttentionRefinement, FeatureFusion
from torch import optim
from tqdm import tqdm

import logging
#
model_file = "D:/Tea/BiSeNet/BiSeNet/source/pytorch-model/resnet18_v1.pth"
if isinstance(model_file, str):
    state_dict = torch.load(model_file, map_location=torch.device('cpu'))
    print(state_dict.keys())
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k
        new_state_dict[name] = v
    # print('new_state_dict:',new_state_dict.keys())
model = ResNet(BasicBlock, [2, 2, 2, 2])
model.load_state_dict(state_dict, strict=False)
# print('ResNet module的关键字',model.state_dict().keys())
print(model.state_dict().keys())
ckpt_keys = set(state_dict.keys())
own_keys = set(model.state_dict().keys())
missing_keys = own_keys - ckpt_keys
print('missing keys', missing_keys)
unexpected_keys = ckpt_keys - own_keys
print('unexpected_keys:', unexpected_keys)

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
    print("t_ioend: ", time.time())

    if is_restore:
        new_state_dict = OrderedDict()
        print('new_state_dict:', new_state_dict.keys(), new_state_dict.values())
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

    def __init__(self, out_planes, is_training,
                 criterion, pretrained_model=None,
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


        self.is_training = is_training

        self.spatial_path = SpatialPath(3, 128, norm_layer)

        conv_channel = 128
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(512, conv_channel, 1, 1, 0,
                       has_bn=True,
                       has_relu=True, has_bias=False, norm_layer=norm_layer)
        )

        # stage = [512, 256, 128, 64]
        arms = [AttentionRefinement(512, conv_channel, norm_layer),
                AttentionRefinement(256, conv_channel, norm_layer)]
        refines = [ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False),
                   ConvBnRelu(conv_channel, conv_channel, 3, 1, 1,
                              has_bn=True, norm_layer=norm_layer,
                              has_relu=True, has_bias=False)]

        heads = [BiSeNetHead(conv_channel, out_planes, 16,
                             True, norm_layer),
                 BiSeNetHead(conv_channel, out_planes, 8,
                             True, norm_layer),
                 BiSeNetHead(conv_channel * 2, out_planes, 8,
                             False, norm_layer)]

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2,
                                 1, norm_layer)

        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)




        if is_training:
            self.criterion = criterion

    def forward(self, data, label=None):
        spatial_out = self.spatial_path(data)

        context_blocks = self.context_path(data)
        context_blocks.reverse()

        global_context = self.global_context(context_blocks[0])
        global_context = F.interpolate(global_context,
                                       size=context_blocks[0].size()[2:],
                                       mode='bilinear', align_corners=True)

        last_fm = global_context
        pred_out = []

        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms,
                                                  self.refines)):
            fm = arm(fm)
            fm += last_fm
            last_fm = F.interpolate(fm, size=(context_blocks[i + 1].size()[2:]),
                                    mode='bilinear', align_corners=True)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)
        # concate_fm = self.heads[-1](concate_fm)
        pred_out.append(concate_fm)

        # if self.is_training:
        #     aux_loss0 = self.criterion(self.heads[0](pred_out[0]), label)
        #     aux_loss1 = self.criterion(self.heads[1](pred_out[1]), label)
        #     main_loss = self.criterion(self.heads[-1](pred_out[2]), label)
        #
        #     loss = main_loss + aux_loss0 + aux_loss1
        #     return loss

        return F.log_softmax(self.heads[-1](pred_out[-1]), dim=1)


class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output


class BiSeNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale,
                 is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 256, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1,
                                       has_bn=True, norm_layer=norm_layer,
                                       has_relu=True, has_bias=False)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(256, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1,
                                      stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        output = self.conv_1x1(fm)
        if self.scale > 1:
            output = F.interpolate(output, scale_factor=self.scale,
                                   mode='bilinear',
                                   align_corners=True)

        return output



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model = 'D:/Tea/BiSeNet/BiSeNet/source/pytorch-model/resnet18_v1.pth'
# 对img的数据处理
full_img = "D:/Tea/CAMUS/training/pictures_3/patient0001_2CH_ED.jpg"
full_img = Image.open(full_img)
print(type(full_img))
scale_factor = 0.5
img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))
img = img.unsqueeze(0)
img = img.to(device=device, dtype=torch.float32)
# 对true_mask的数据处理
true_mask = "D:/Tea/CAMUS/training/masks_6/patient0001_2CH_ED_gt.png"
true_mask = Image.open(true_mask)
true_mask = torch.from_numpy(BasicDataset.preprocess(true_mask, scale_factor))
true_mask = true_mask.unsqueeze(0)
# true_mask=true_mask.squeeze()
true_mask = true_mask.to(device=device, dtype=torch.float64)
true_mask = F.interpolate(true_mask, size=(392, 280),
                              mode='bilinear',
                              align_corners=True)
true_mask = true_mask.to(device=device, dtype=torch.long)
true_mask=torch.squeeze(true_mask)
# true_mask=torch.squeeze(true_mask)
true_mask=true_mask.unsqueeze(0)


num_classes = 4
batch_size = 16
image_height = 778
image_width = 549

criterion = nn.CrossEntropyLoss()
pretrained_model = "D:/Tea/BiSeNet/BiSeNet/source/pytorch-model/resnet18_v1.pth"
net = BiSeNet(out_planes=num_classes, is_training=True,
              criterion=criterion,
              pretrained_model=pretrained_model,
              )
net.to(device=device)
# pred_mask = net(img)
lr = 0.001
optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
sum=10
dir_checkpoint="Single-Picture-Context/"
try:
    os.mkdir(dir_checkpoint)
    logging.info('Created checkpoint directory')
except OSError:
    pass
for epoch in range(10):
    with tqdm(total=1, desc=f'Epoch {epoch + 1}/{sum}', unit='img') as pbar:
        net.train()
        optimizer.zero_grad()

        masks_pred = net(img)

        print(masks_pred.shape)
        print(true_mask.shape)
        print(masks_pred.size()[2:])
        print('the type of true_mask:',type(true_mask))





        loss = criterion(masks_pred, true_mask)
        loss.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()

        torch.save(net.state_dict(),
                   dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
        logging.info(f'Checkpoint {epoch + 1} saved !')
