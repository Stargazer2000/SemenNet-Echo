import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os.path



class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        print("这是进行view变换后的shape:")
        print((input.view(-1)).shape)
        print((target.view(-1)).shape)
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

class MultiDiceCoeff(Function):
    def forward(self,input,target):
        eps=1e-7
        # sum=0
        # mean=0
        # list=[]
        # print('input.shape',input.shape[0])
        # for i in range(input.shape[0]):
        #     self.inter = torch.dot(input[i].view(-1), target[i].view(-1))
        #     self.union = torch.sum(input[i]) + torch.sum(target[i]) + eps
        #     # list[i]=(2 * self.inter.float() + eps) / self.union.float()
        #     list.append((2 * self .inter.float() + eps) / self.union.float())
        # return list
        # input = (input> 0.5).float()
        #
        # dims = (0,) + tuple(range(2, target.ndimension()))
        # inter= (input * target).sum(axis=[-1, -2])
        # union=(input + target).sum(axis=[-1, -2])
        # dice = (2. * inter / (union + eps)).mean(axis=-1)
        SumDice = 0
        for i in range(0, input.shape[0]):
            input[i] = (input[i] > 0.5).float()
            inter = torch.dot(input[i].view(-1), target[i].view(-1))
            union = torch.sum(input[i]) + torch.sum(target[i]) + eps
            # list[i]=(2 * self.inter.float() + eps) / self.union.float()
            # list.append((2 * inter.float() + eps) / union.float())
            dice=(2 * inter.float() + eps) / union.float()
            SumDice+=dice
        return SumDice/4




def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)



def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        # i+1是validation set的batch_size
        print('i in enumerate:',i)
        # s = s + MultiDiceCoeff().forward(c[0], c[1])
        print('c0:',c[0].shape)
        print('c1:',c[1].shape)
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)


def multi_dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    s = 0


    for i, c in enumerate(zip(input, target)):
        pred=c[0].cpu()
        mask=c[1].cpu()
        mask=get_one_hot(mask,4)

        mask=mask.permute(2,0,1)
        # print('mask:',mask.shape)
        # print('pred:',pred.shape)
        softmax_func = nn.Softmax(dim=0)

        pred_mask = softmax_func(pred)
        s+=MultiDiceCoeff().forward(pred_mask, mask)

    return s/(i+1)
        #加上背景总共分4类，左心室内膜、外膜、左心房，一共有3个channel，分别获取每一个类别的dice



        # 将list2中的每个值求平均，求得在这个batch中的每一个类别的平均dice

        # s = s + MultiDiceCoeff().forward(c[0], c[1])
def PictureTransform(output):
    probs = torch.sigmoid(output)
    probs = probs.squeeze(0)
    tf = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((549,778)),
            transforms.ToTensor()
        ]
    )
    probs = tf(probs.cpu())
    full_mask = probs.squeeze().cpu().numpy()

    return full_mask

def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))



def dice_list(input,target):

    eps=1e-7
    # print('input.shape',input.shape[0])
    list=[]
    for i in range(0,input.shape[0]):
        input[i]=(input[i]>0.5).float()
        inter = torch.dot(input[i].view(-1), target[i].view(-1))
        union = torch.sum(input[i]) + torch.sum(target[i]) + eps
        # list[i]=(2 * self.inter.float() + eps) / self.union.float()
        list.append((2 * inter.float() + eps) / union.float())
    return list

# precision: TP/(TP+FP)
def precision_list(input,target):
    eps = 1e-7
    # print('input.shape',input.shape[0])
    list = []
    for i in range(0, input.shape[0]):
        input[i] = (input[i] > 0.5).float()
        inter=torch.dot(input[i].view(-1), target[i].view(-1))
        # positive=TP+FP
        positive=inter+torch.sum((input[i]==1)&(target[i]==0))+ eps
        # list[i]=(2 * self.inter.float() + eps) / self.union.float()
        list.append(inter.float() / positive.float())
    return list

def jacc_list(input,target):
    eps=1e-7
    list=[]
    for i in range(0,input.shape[0]):
        input[i] = (input[i] > 0.5).float()
        inter = torch.dot(input[i].view(-1), target[i].view(-1))
        union = torch.sum(input[i]) + torch.sum(target[i])-inter + eps
        list.append(inter.float()/union.float())
    return list

def accuracy_list(input,target):
    eps=1e-7
    list=[]
    for i in range(0,input.shape[0]):
        input[i] = (input[i] > 0.5).float()
        # TP+TN
        true=torch.sum(input[i].view(-1)==target[i].view(-1))
        # TP+FP+FN+TN
        total=input[i].shape[-1]*input[i].shape[-2]+eps
        list.append((true+eps)/(total))
    return list
# specificity=TN/(TN+FP)
def specificity_list(input,target):
    eps=1e-7
    list=[]
    for i in range(0,input.shape[0]):
        input[i] = (input[i] > 0.5).float()
        # TN
        TN=torch.sum((input[i]==0)&(target[i]==0))
        # negative=TN+FP
        FP=torch.sum((input[i]==1)&(target[i]==0))+eps
        list.append(TN/(TN+FP))
    return list


m=0
def multi_dice_coeff_list(input, target, epochNum):
    global i
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    list2=[0,0,0,0]


    for i, c in enumerate(zip(input, target)):
        # i+1是validation set的batch_size
        pred = c[0].cpu()
        mask = c[1].cpu()
        mask = get_one_hot(mask, 4)

        mask = mask.permute(2, 0, 1)
        softmax_func = nn.Softmax(dim=0)
        pred_mask = softmax_func(pred)
        # print('pred_mask before transform:',pred_mask.shape)
        list=dice_list(pred_mask,mask)
        # print(list)

        outdir='./images/'

        mask111=PictureTransform(pred_mask[0])
        result=mask_to_image(mask111)
        # result.save(os.path.join(outdir,'epoch'+str(epochNum)+'_'+str(++i)+'_0.jpg'))
        # print(os.path.basename(c[0].cpu()))
        result.save('./transform0.jpg')

        outdir1='./images1/'
        true=PictureTransform(pred_mask[1])
        result01 = mask_to_image(true)
        # result01.save(os.path.join(outdir1,'epoch'+str(epochNum)+'_'+str(++i)+'_1.jpg'))
        result01.save('./transform1.jpg')


        outdir2='./images2/'
        true = PictureTransform(pred_mask[2])
        result01 = mask_to_image(true)
        # result01.save(os.path.join(outdir2,'epoch'+str(epochNum)+'_'+str(++i)+'_2.jpg'))
        result01.save('./transform2.jpg')

        outdir3='./images3/'
        true = PictureTransform(pred_mask[3])
        result01 = mask_to_image(true)
        # result01.save(os.path.join(outdir3,'epoch'+str(epochNum)+'_'+str(++i)+'_3.jpg'))
        result01.save('./transform3.jpg')

        list2[0]+=list[0]
        list2[1]+=list[1]
        list2[2]+=list[2]
        list2[3]+=list[3]
    list2[0],list2[1],list2[2],list2[3]=list2[0]/(i+1),list2[1]/(i+1),list2[2]/(i+1),list2[3]/(i+1)
    print('Dice  list2[0],list2[1],list2[2],list2[3]:',list2[0],list2[1],list2[2],list2[3])




    return list2


# 补充评判标准dm,dh,accuracy,jacc,precision,specificity
# jacc
def jacc_coeff_list(input,target):
    global i
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    list2 = [0, 0, 0, 0]

    for i, c in enumerate(zip(input, target)):
        # i+1是validation set的batch_size
        pred = c[0].cpu()
        mask = c[1].cpu()
        mask = get_one_hot(mask, 4)

        mask = mask.permute(2, 0, 1)
        softmax_func = nn.Softmax(dim=0)
        pred_mask = softmax_func(pred)
        # print('pred_mask before transform:',pred_mask.shape)
        list = jacc_list(pred_mask, mask)
        list2[0] += list[0]
        list2[1] += list[1]
        list2[2] += list[2]
        list2[3] += list[3]
    list2[0], list2[1], list2[2], list2[3] = list2[0] / (i + 1), list2[1] / (i + 1), list2[2] / (i + 1), list2[3] / (
                i + 1)
    print('Jacc list2[0],list2[1],list2[2],list2[3]:', list2[0], list2[1], list2[2], list2[3])

    return list2

# accuracy
def accuracy_coeff_list(input,target):
    global i
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    list2 = [0, 0, 0, 0]

    for i, c in enumerate(zip(input, target)):
        # i+1是validation set的batch_size
        pred = c[0].cpu()
        mask = c[1].cpu()
        mask = get_one_hot(mask, 4)

        mask = mask.permute(2, 0, 1)
        softmax_func = nn.Softmax(dim=0)
        pred_mask = softmax_func(pred)
        # print('pred_mask before transform:',pred_mask.shape)
        list = accuracy_list(pred_mask, mask)
        list2[0] += list[0]
        list2[1] += list[1]
        list2[2] += list[2]
        list2[3] += list[3]
    list2[0], list2[1], list2[2], list2[3] = list2[0] / (i + 1), list2[1] / (i + 1), list2[2] / (i + 1), list2[3] / (
                i + 1)
    print('Accuracy  list2[0],list2[1],list2[2],list2[3]:', list2[0], list2[1], list2[2], list2[3])

    return list2

# precision
def precision_coeff_list(input,target):
    global i
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    list2 = [0, 0, 0, 0]

    for i, c in enumerate(zip(input, target)):
        # i+1是validation set的batch_size
        pred = c[0].cpu()
        mask = c[1].cpu()
        mask = get_one_hot(mask, 4)

        mask = mask.permute(2, 0, 1)
        softmax_func = nn.Softmax(dim=0)
        pred_mask = softmax_func(pred)
        # print('pred_mask before transform:',pred_mask.shape)
        list = precision_list(pred_mask, mask)
        list2[0] += list[0]
        list2[1] += list[1]
        list2[2] += list[2]
        list2[3] += list[3]
    list2[0], list2[1], list2[2], list2[3] = list2[0] / (i + 1), list2[1] / (i + 1), list2[2] / (i + 1), list2[3] / (
                i + 1)
    print('Precision: list2[0],list2[1],list2[2],list2[3]:', list2[0], list2[1], list2[2], list2[3])

    return list2


# specificity

def specificity_coeff_list(input,target):
    global i
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    list2 = [0, 0, 0, 0]

    for i, c in enumerate(zip(input, target)):
        # i+1是validation set的batch_size
        pred = c[0].cpu()
        mask = c[1].cpu()
        mask = get_one_hot(mask, 4)

        mask = mask.permute(2, 0, 1)
        softmax_func = nn.Softmax(dim=0)
        pred_mask = softmax_func(pred)
        # print('pred_mask before transform:',pred_mask.shape)
        list = specificity_list(pred_mask, mask)
        list2[0] += list[0]
        list2[1] += list[1]
        list2[2] += list[2]
        list2[3] += list[3]
    list2[0], list2[1], list2[2], list2[3] = list2[0] / (i + 1), list2[1] / (i + 1), list2[2] / (i + 1), list2[3] / (
                i + 1)
    print('Specificity  list2[0],list2[1],list2[2],list2[3]:', list2[0], list2[1], list2[2], list2[3])

    return list2


def test_dice_coeff_list(input, target):
    global i
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    list2=[0,0,0,0]


    for i, c in enumerate(zip(input, target)):
        # i+1是validation set的batch_size
        pred = c[0].cpu()
        mask = c[1].cpu()
        mask = get_one_hot(mask, 4)

        mask = mask.permute(2, 0, 1)
        softmax_func = nn.Softmax(dim=0)
        pred_mask = softmax_func(pred)
        # print('pred_mask before transform:',pred_mask.shape)
        list=dice_list(pred_mask,mask)
        # print(list)

        outdir='./images/'

        mask111=PictureTransform(pred_mask[0])
        result=mask_to_image(mask111)
        # result.save(os.path.join(outdir,'epoch'+str(epochNum)+'_'+str(++i)+'_0.jpg'))
        # print(os.path.basename(c[0].cpu()))
        result.save('./transform0.jpg')

        outdir1='./images1/'
        true=PictureTransform(pred_mask[1])
        result01 = mask_to_image(true)
        # result01.save(os.path.join(outdir1,'epoch'+str(epochNum)+'_'+str(++i)+'_1.jpg'))
        result01.save('./transform1.jpg')


        outdir2='./images2/'
        true = PictureTransform(pred_mask[2])
        result01 = mask_to_image(true)
        # result01.save(os.path.join(outdir2,'epoch'+str(epochNum)+'_'+str(++i)+'_2.jpg'))
        result01.save('./transform2.jpg')

        outdir3='./images3/'
        true = PictureTransform(pred_mask[3])
        result01 = mask_to_image(true)
        # result01.save(os.path.join(outdir3,'epoch'+str(epochNum)+'_'+str(++i)+'_3.jpg'))
        result01.save('./transform3.jpg')

        list2[0]+=list[0]
        list2[1]+=list[1]
        list2[2]+=list[2]
        list2[3]+=list[3]
    list2[0],list2[1],list2[2],list2[3]=list2[0]/(i+1),list2[1]/(i+1),list2[2]/(i+1),list2[3]/(i+1)
    print('Dice:  list2[0],list2[1],list2[2],list2[3]:',list2[0],list2[1],list2[2],list2[3])




    return list2












