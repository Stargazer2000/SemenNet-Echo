# encoding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
# from torchvision.models import resnet50, resnet101, resnet152

import sys
sys.path.append("./BiSeNet")
from BiSeNet.resnet import resnet18
from BiSeNet.seg_oprs import ConvBnRelu, AttentionRefinement, FeatureFusion

num_classes=4
def get():
    return BiSeNet(num_classes, None,None)


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
        # self.spatial_path_second=SpatialPathSecond(3,128,norm_layer)
        # self.spatial_path_third=SpatialPathThird(3,128,norm_layer)

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
        # 在这个head的倒数第二个添加了，scale=4的上采样

        heads = [BiSeNetHead(conv_channel, out_planes, 16,
                             True, norm_layer),
                 BiSeNetHead(conv_channel, out_planes, 8,
                             True, norm_layer),
                 BiSeNetHead(256, out_planes, 4,
                             True, norm_layer),
                 BiSeNetHead(conv_channel * 2, out_planes, 8,
                             False, norm_layer)]

        self.ffm = FeatureFusion(conv_channel * 2, conv_channel * 2,
                                 1, norm_layer)
        self.ffm2=FeatureFusion(128,conv_channel * 2,1,norm_layer)


        self.arms = nn.ModuleList(arms)
        self.refines = nn.ModuleList(refines)
        self.heads = nn.ModuleList(heads)


        self.business_layer.append(self.spatial_path)
        self.business_layer.append(self.global_context)
        self.business_layer.append(self.arms)
        self.business_layer.append(self.refines)
        self.business_layer.append(self.heads)
        self.business_layer.append(self.ffm)

        if is_training:
            self.criterion = criterion

    def forward(self, data, label=None):
        spatial_out = self.spatial_path(data)['third']
        spatial_out_second=self.spatial_path(data)['second']
        spatial_out_third=self.spatial_path(data)['first']

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

        # 将feature map大小为39*39和大小为77*77的特征图分别concat后，得到了两个concat后的特征图。
        # 将大小为39*39的特征图interpolate到77*77后，相加，相加后的结果再interpolate 4倍，mask进行padding处理后即可进行loss的计算（先融合两个特征图看下结果）
        # 关于loss的改进，将相加融合前的特征图分别上采样和mask相同的大小，loss=loss1+loss2+loss3
        # 以下代码是concat、相加的改进
        #       last_fm 39*39  concat_fm1 39*39    concat_fm2: 77*77,
        concat_fm1=self.ffm(spatial_out,last_fm)
        print('spatial_out_second,context_blocks[3]',spatial_out_second.shape,context_blocks[3].shape)

        concat_fm2=self.ffm2(spatial_out_second,context_blocks[3])
        last_fm_context=F.interpolate(context_blocks[3],size=(spatial_out_third.size()[2:]),mode='bilinear',align_corners=True)
        concat_fm3 = self.ffm2(spatial_out_third,last_fm_context)
        #       将concat_fm1上采样到77*77，再相加
        concat_fm1=F.interpolate(concat_fm1,size=(concat_fm3.size()[2:]),mode='bilinear',align_corners=True)
        concat_fm2 = F.interpolate(concat_fm2, size=(concat_fm3.size()[2:]), mode='bilinear', align_corners=True)

        concat=concat_fm2+concat_fm1+concat_fm3
        print('concat:channel',concat.shape)
        # 最后一步，将输出后的结果经过4倍上采样后输出
        # 下面是将合并后的77*77大小的feature map上采样到154*154，再和154*154大小的feature map融合最后经过2倍上采样
        # print('concat_fm3:',concat_fm3.shape)
        concat_final=concat/3




        # 以上代码是concat、相加的改进

        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)
        # concate_fm = self.ffm(context_out, context_out)
        # concate_fm = self.ffm(spatial_out, spatial_out)
        pred_out.append(concate_fm)

        # if self.is_training:
        #     aux_loss0 = self.criterion(self.heads[0](pred_out[0]), label)
        #     aux_loss1 = self.criterion(self.heads[1](pred_out[1]), label)
        #     main_loss = self.criterion(self.heads[-1](pred_out[2]), label)
        #
        #     loss = main_loss + aux_loss0 + aux_loss1
        #     return loss

        # return F.log_softmax(self.heads[-1](pred_out[-1]), dim=1)
        return F.log_softmax(self.heads[-2](concat_final), dim=1)




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
        x1 = self.conv_3x3_1(x)
        x2 = self.conv_3x3_2(x1)
        output = self.conv_1x1(x2)

        return {
            'first':x,
            'second':x1,
            'third':output
        }


class SpatialPathSecond(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPathSecond, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        # self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
        #                              has_bn=True, norm_layer=norm_layer,
        #                              has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        # x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output


class SpatialPathThird(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPathThird, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        # self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
        #                              has_bn=True, norm_layer=norm_layer,
        #                              has_relu=True, has_bias=False)
        # self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
        #                              has_bn=True, norm_layer=norm_layer,
        #                              has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        # x = self.conv_3x3_1(x)
        # x = self.conv_3x3_2(x)
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


# if __name__ == "__main__":
#     criterion=nn.CrossEntropyLoss()
#     model = BiSeNet(4, None,criterion=criterion)
#     print(model)
