# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
# from torchvision.models import resnet50, resnet101, resnet152
from CascadePSP.models.sync_batchnorm import SynchronizedBatchNorm2d

import sys
sys.path.append("./BiSeNet")
from BiSeNet.resnet import resnet18
from BiSeNet.seg_oprs import ConvBnRelu, AttentionRefinement, FeatureFusion

num_classes=4
def get():
    return BiSeNet(num_classes, None,None)

class DownSample(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        set_priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages]
        priors = set_priors + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class PSPUpsample(nn.Module):
    def __init__(self, x_channels, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            SynchronizedBatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            SynchronizedBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.conv2 = nn.Sequential(
            SynchronizedBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            SynchronizedBatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

        self.shortcut = nn.Conv2d(x_channels, out_channels, kernel_size=1)

    def forward(self, x, up):
        x = F.interpolate(input=x, scale_factor=2, mode='bilinear', align_corners=False)

        p = self.conv(torch.cat([x, up], 1))
        sc = self.shortcut(x)

        p = p + sc

        p2 = self.conv2(p)

        return p + p2


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

        self.business_layer = []
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


        self.business_layer.append(self.spatial_path)
        self.business_layer.append(self.global_context)
        self.business_layer.append(self.arms)
        self.business_layer.append(self.refines)
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)

        self.psp=DownSample(512,1024,(1,2,3,6))
        # 以下通道数需要进行修改
        self.up_1 = PSPUpsample(1024, 1024 + 256, 512)
        self.up_2 = PSPUpsample(512, 512 + 64, 256)
        self.up_3 = PSPUpsample(256, 256 + 3, 32)
        self.final_28 = nn.Sequential(
            nn.Conv2d(1024, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self.final_56 = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        self.final_11 = nn.Conv2d(32 + 3, 32, kernel_size=1)
        self.final_21 = nn.Conv2d(32, 1, kernel_size=1)

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
        # concate_fm = self.ffm(context_out, context_out)
        # concate_fm = self.ffm(spatial_out, spatial_out)
        pred_out.append(concate_fm)
        seg=self.heads[-1](pred_out[-1])
        diffY = seg.size()[2] - data.size()[2]
        diffX = seg.size()[3] - data.size()[3]

        img = F.pad(data, [diffX // 2, diffX - diffX // 2,
                                        diffY // 2, diffY - diffY // 2])
        p=torch.cat((img,seg,seg,seg),1)
        # concat后是和data相同的大小，使用resnet18进行特征提取，
        p_blocks=self.context_path(p)
        p_blocks.reverse()
        # 此时最后一层p_blocks的channel数应为：512
        # 将池化后的模块上采样到resnet18特征提取后的大小，
        p=self.psp(p_blocks[0])
        # 下面代码中upsample的通道数需要统一修改
        inter_s8 = self.final_28(p)
        # 下行代码扩大的倍数需要修改
        r_inter_s8 = F.interpolate(inter_s8, scale_factor=8, mode='bilinear', align_corners=False)
        r_inter_tanh_s8 = torch.tanh(r_inter_s8)
        images={}

        images['pred_28'] = torch.sigmoid(r_inter_s8)
        images['out_28'] = r_inter_s8
        # the second step

        p = torch.cat((img, seg, r_inter_tanh_s8, r_inter_tanh_s8), 1)
        p_blocks = self.context_path(p)
        p_blocks.reverse()
        p,f_1,f_2=p_blocks[0]
        p = self.psp(p_blocks[0])
        inter_s8_2 = self.final_28(p)
        r_inter_s8_2 = F.interpolate(inter_s8_2, scale_factor=8, mode='bilinear', align_corners=False)
        r_inter_tanh_s8_2 = torch.tanh(r_inter_s8_2)

        p = self.up_1(p, f_2)

        inter_s4 = self.final_56(p)
        r_inter_s4 = F.interpolate(inter_s4, scale_factor=4, mode='bilinear', align_corners=False)
        r_inter_tanh_s4 = torch.tanh(r_inter_s4)

        images['pred_28_2'] = torch.sigmoid(r_inter_s8_2)
        images['out_28_2'] = r_inter_s8_2
        images['pred_56'] = torch.sigmoid(r_inter_s4)
        images['out_56'] = r_inter_s4

        # the last step
        p = torch.cat((img, seg, r_inter_tanh_s8_2, r_inter_tanh_s4), 1)
        p_blocks = self.context_path(p)
        p_blocks.reverse()
        p, f_1, f_2 = p_blocks[0]
        p = self.psp(p_blocks[0])
        inter_s8_3 = self.final_28(p)
        r_inter_s8_3 = F.interpolate(inter_s8_3, scale_factor=8, mode='bilinear', align_corners=False)

        p = self.up_1(p, f_2)
        inter_s4_2 = self.final_56(p)
        r_inter_s4_2 = F.interpolate(inter_s4_2, scale_factor=4, mode='bilinear', align_corners=False)
        p = self.up_2(p, f_1)
        p = self.up_3(p, data)

        """
        Final output,注意output的通道数修改
        """
        p = F.relu(self.final_11(torch.cat([p, data], 1)), inplace=True)
        p = self.final_21(p)

        pred_224 = torch.sigmoid(p)

        images['pred_224'] = pred_224
        images['out_224'] = p
        images['pred_28_3'] = torch.sigmoid(r_inter_s8_3)
        images['pred_56_2'] = torch.sigmoid(r_inter_s4_2)
        images['out_28_3'] = r_inter_s8_3
        images['out_56_2'] = r_inter_s4_2












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
        x1 = self.conv_7x7(x)
        x2 = self.conv_3x3_1(x1)
        x3 = self.conv_3x3_2(x2)
        output = self.conv_1x1(x3)

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


# if __name__ == "__main__":
#     criterion=nn.CrossEntropyLoss()
#     model = BiSeNet(4, None,criterion=criterion)
#     print(model)
