import torch
from torch.nn import functional as f

# x = torch.arange(0, 1 * 3 * 4*4).float()
# x = x.view(1, 3, 4,4)
# print(x)
# x1 = f.unfold(x, kernel_size=2, dilation=1, stride=1)
# print(x1.shape)
# B, C_kh_kw, L = x1.size()
# x1 = x1.permute(0, 2, 1)
# x1 = x1.view(B, L, -1, 2, 2)
# print(x1)
#
#
# y= torch.arange(0, 1 * 3 * 4*4).float()
# y = y.view(1, 3, 4, 4)
# print(y)
# y=y.view(1,3,2,2,2,2)
# y=y.permute(0,2,4,1,3,5).contiguous().view(-1, 3, 2, 2)
# print(y)
import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from torch.nn import init
from torch.nn.parameter import Parameter

class ExternalAttention(nn.Module):

    def __init__(self, d_model,S=32,G=8):
        super().__init__()
        d_model=d_model//G
        self.mk=nn.Linear(d_model,S,bias=False)
        self.mv=nn.Linear(S,d_model,bias=False)
        self.softmax=nn.Softmax(dim=2)
        self.G=G
        self.init_weights()
        self.sweight = Parameter(torch.zeros(1, 32, 1, 1))
        self.sbias = Parameter(torch.ones(1, 32, 1, 1))
        self.gn = nn.GroupNorm(32, 32)
        self.sigmoid=nn.Sigmoid()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries):
        # 对attn进行分group
        b,h,w,c=queries.shape
        queries=queries.reshape(b,h,w,self.G,-1)
        queries=queries.permute(0,3,1,2,4)
        b0,g0,h0,w0,c0=queries.shape
        queries=queries.reshape(-1,h0,w0,c0)


        attn=self.mk(queries) #bs,n,S
        # attn=self.softmax(attn) #bs,n,S
        # attn=attn/torch.sum(attn,dim=3,keepdim=True) #bs,n,S
        # 修改正则化的方式，将其修改为group Norm的方式,此时[b,h,w,c],通道数为32，d_model是64,d_model*group恢复原通道数，恢复形式为[b,h,w,c]的形式
        attn=attn.permute(0,3,1,2)#[b,c,h,w]
        attn_spatial=self.gn(attn)
        attn_spatial=self.sweight*attn_spatial+self.sbias #bs*G,c//(2*G),h,w
        attn=attn*self.sigmoid(attn_spatial) #bs*G,c//(2*G),h,w
        attn=attn.permute(0,2,3,1)
        attn=self.mv(attn)
        attn=attn.reshape(b,self.G,h,w,-1)
        out=attn.permute(0,2,3,1,4)
        out=out.contiguous().view(b,h,w,-1)


        return out


class SelfAttentionBlock2D(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''

    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, bn_type=None):
        super(SelfAttentionBlock2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        if out_channels == None:
            self.out_channels = in_channels
        self.f_key = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(self.in_channels, self.key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.key_channels, self.key_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(key_channels),
            nn.ReLU(inplace=True),
        )

        self.f_value = nn.Conv2d(self.in_channels, self.value_channels, kernel_size=1, bias=False)
        self.W = nn.Sequential(
            nn.Conv2d(self.value_channels, self.out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)

        value = self.f_value(x).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)
        query = self.f_query(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_key(x).view(batch_size, self.key_channels, -1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        context = self.W(context)
        return context


class ISA_Block(nn.Module):
    def __init__(self,in_channels,out_channels, down_factor=[2, 2], bn_type=None):
        super(ISA_Block, self).__init__()
        self.out_channels = out_channels
        assert isinstance(down_factor, (tuple, list)) and len(down_factor) == 2
        self.down_factor = down_factor
        # self.long_range_sa = ExternalAttention(d_model=512,S=64)
        # self.short_range_sa = ExternalAttention(d_model=64,S=32)#之前用的是这个
        self.short_range_sa = ExternalAttention(d_model=in_channels, S=32)


    def forward(self, x):
        n, c, h, w = x.size()
        dh, dw = self.down_factor  # down_factor for h and w, respectively

        out_h, out_w = math.ceil(h / dh), math.ceil(w / dw)
        # pad the feature if the size is not divisible
        pad_h, pad_w = out_h * dh - h, out_w * dw - w
        if pad_h > 0 or pad_w > 0:  # padding in both left&right sides
            feats = F.pad(x, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            feats = x

        # long range attention
        feats = feats.view(n, c, out_h, dh, out_w, dw)
        feats = feats.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, c, dh, dw)
        feats = feats.permute(0,2,3,1)
        print(feats.shape)
        feats = self.short_range_sa(feats)
        c = self.out_channels
        # 因为去掉了long-range，所以 以下代码不要

        # short range attention
        # feats = feats.view(n, dh, dw, c, out_h, out_w)
        # feats = feats.permute(0, 4, 5, 3, 1, 2).contiguous().view(-1, c, dh, dw)
        # feats=feats.permute(0,2,3,1)
        # print(feats.shape)
        # feats = self.short_range_sa(feats)
        # 以上代码不需要
        feats = feats.permute(0,3,1,2)
        feats = feats.view(n, out_h, out_w, c, dh, dw).permute(0, 3, 1, 4, 2, 5)
        feats = feats.contiguous().view(n, c, dh * out_h, dw * out_w)

        # remove padding
        if pad_h > 0 or pad_w > 0:
            feats = feats[:, :, pad_h // 2:pad_h // 2 + h, pad_w // 2:pad_w // 2 + w]
        feats=feats+x

        return feats


a=torch.randn([24,2048,7,7])
selfAttn=ISA_Block(in_channels=2048,out_channels=2048)
print(selfAttn(a).size())