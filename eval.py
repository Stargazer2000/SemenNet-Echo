import torch
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
from dice_loss import dice_coeff,multi_dice_coeff,multi_dice_coeff_list,test_dice_coeff_list,accuracy_coeff_list,precision_coeff_list,jacc_coeff_list,specificity_coeff_list


def eval_net(net, loader, device, epochNum ):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    # mask_type = torch.float32 if net.n_classes == 1 else torch.long
    mask_type=torch.long
    n_val = len(loader)  # the number of batch
    # print('n_val:',n_val)
    tot = 0
    list2=[0,0,0,0]
    list3=[0,0,0,0]
    list4=[0,0,0,0]
    list5=[0,0,0,0]
    list6=[0,0,0,0]


    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            # print('true_masks:',true_masks.shape)

            with torch.no_grad():
                mask_pred = net(imgs)
                mask_pred=torch.squeeze(mask_pred)

                # true_masks = true_masks.to(device=device, dtype=torch.float64)
                # true_masks.unsqueeze(0)
                # true_masks = F.interpolate(true_masks, size=(mask_pred.shape[2:]),mode='bilinear', align_corners=True)
                # print('padding之前true_masks和masks_pred的shape', true_masks.shape, mask_pred.shape)
                diffY = mask_pred.size()[2] - true_masks.size()[2]
                diffX = mask_pred.size()[3] - true_masks.size()[3]

                true_masks = F.pad(true_masks, [diffX // 2, diffX - diffX // 2,
                                                diffY // 2, diffY - diffY // 2])
                # print('padding之后true_masks和masks_pred的shape', true_masks.shape, mask_pred.shape)

                true_masks = true_masks.to(device=device, dtype=mask_type)
                true_masks = torch.squeeze(true_masks, dim=1)
                # print('mask_pred:',mask_pred.shape)
                # print('true_mask:',true_masks.shape)

            if net.n_classes > 1:
                tot+=multi_dice_coeff(mask_pred,true_masks)
                list=multi_dice_coeff_list(mask_pred,true_masks,epochNum)
                # list=test_dice_coeff_list(mask_pred,true_masks)
                listAccuracy=accuracy_coeff_list(mask_pred,true_masks)
                listPrecision=precision_coeff_list(mask_pred,true_masks)
                listJacc=jacc_coeff_list(mask_pred,true_masks)
                listSpecificity=specificity_coeff_list(mask_pred,true_masks)

                list2[0] += list[0]
                list2[1] += list[1]
                list2[2] += list[2]
                list2[3] += list[3]

                list3[0] += listAccuracy[0]
                list3[1] += listAccuracy[1]
                list3[2] += listAccuracy[2]
                list3[3] += listAccuracy[3]

                list4[0] += listPrecision[0]
                list4[1] += listPrecision[1]
                list4[2] += listPrecision[2]
                list4[3] += listPrecision[3]

                list5[0]+=listJacc[0]
                list5[1]+=listJacc[1]
                list5[2]+=listJacc[2]
                list5[3]+=listJacc[3]

                list6[0]+=listSpecificity[0]
                list6[1]+=listSpecificity[1]
                list6[2]+=listSpecificity[2]
                list6[3]+=listSpecificity[3]


            # tot += F.cross_entropy(mask_pred, true_masks).item()

            # 第二个是原版tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                # print('pred:',pred.shape)
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    tot=tot/n_val
    list2[0], list2[1], list2[2], list2[3] = list2[0] /n_val, list2[1] / n_val, list2[2] / n_val, list2[3] / n_val
    list3[0], list3[1], list3[2], list3[3] = list3[0] / n_val, list3[1] / n_val, list3[2] / n_val, list3[3] / n_val
    list4[0], list4[1], list4[2], list4[3] = list4[0] / n_val, list4[1] / n_val, list4[2] / n_val, list4[3] / n_val
    list5[0], list5[1], list5[2], list5[3] = list5[0] / n_val, list5[1] / n_val, list5[2] / n_val, list5[3] / n_val
    list6[0], list6[1], list6[2], list6[3] = list6[0] / n_val, list6[1] / n_val, list6[2] / n_val, list6[3] / n_val

    return {
        'mean':tot,
        'list':list2,
        'AccuracyList':list3,
        'Precision':list4,
        'Jacc':list5,
        'Specificity':list6
    }

def test_eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    # print('n_val:',n_val)
    tot = 0
    list2=[0,0,0,0]


    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            # print('true_masks:',true_masks.shape)

            with torch.no_grad():
                mask_pred = net(imgs)
                mask_pred=torch.squeeze(mask_pred)

                # true_masks = true_masks.to(device=device, dtype=torch.float64)
                # true_masks.unsqueeze(0)
                # true_masks = F.interpolate(true_masks, size=(mask_pred.shape[2:]),mode='bilinear', align_corners=True)
                # print('padding之前true_masks和masks_pred的shape', true_masks.shape, mask_pred.shape)
                diffY = mask_pred.size()[2] - true_masks.size()[2]
                diffX = mask_pred.size()[3] - true_masks.size()[3]

                true_masks = F.pad(true_masks, [diffX // 2, diffX - diffX // 2,
                                                diffY // 2, diffY - diffY // 2])
                # print('padding之后true_masks和masks_pred的shape', true_masks.shape, mask_pred.shape)

                true_masks = true_masks.to(device=device, dtype=mask_type)
                true_masks = torch.squeeze(true_masks, dim=1)
                # print('mask_pred:',mask_pred.shape)
                # print('true_mask:',true_masks.shape)

            if net.n_classes > 1:
                tot+=multi_dice_coeff(mask_pred,true_masks)
                # list=multi_dice_coeff_list(mask_pred,true_masks,epochNum)
                list=test_dice_coeff_list(mask_pred,true_masks)
                list2[0] += list[0]
                list2[1] += list[1]
                list2[2] += list[2]
                list2[3] += list[3]


            # tot += F.cross_entropy(mask_pred, true_masks).item()

            # 第二个是原版tot += F.cross_entropy(mask_pred, true_masks).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                # print('pred:',pred.shape)
                tot += dice_coeff(pred, true_masks).item()
            pbar.update()

    net.train()
    tot=tot/n_val
    list2[0], list2[1], list2[2], list2[3] = list2[0] /n_val, list2[1] / n_val, list2[2] / n_val, list2[3] / n_val

    return {
        'mean':tot,
        'list':list2
    }