import argparse
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet
from BiSeNet.network import BiSeNet
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
from BiSeNet.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d
from dice_loss import dice_coeff,multi_dice_coeff,multi_dice_coeff_list
from glob import glob
dir_img='D:/Tea/CAMUS/training/pictures_3/'
dir_mask = 'D:/Tea/CAMUS/training/masks_6/'
# dir_img='D:/Tea/BiSeNet/BiSeNet/img/'
# dir_mask='D:/Tea/BiSeNet/BiSeNet/mask/'
dir_checkpoint = 'SpatialCheckpoints/'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def calculate_dice(input, target):
    eps = 0.0001
    sum = 0
    mean = 0
    list = {}
    for i in range(0,input.shape[1]):
        inter = torch.dot(input[i].view(-1), target[i].view(-1))
        union = torch.sum(input[i]) + torch.sum(target[i]) + eps
        list[i] = (2 * inter.float() + eps) / union.float()

    return list

def get_one_hot(label, N):
    size = list(label.size())
    label = label.view(-1)   # reshape 为向量
    ones = torch.sparse.torch.eye(N)
    ones = ones.index_select(0, label)   # 用上面的办法转为换one hot
    size.append(N)  # 把类别输目添到size的尾后，准备reshape回原来的尺寸
    return ones.view(*size)


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        list3=[0,0,0,0]
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)

                optimizer.zero_grad()

                masks_pred = net(imgs)
                masks_pred=torch.squeeze(masks_pred)

                print('true_mask:',true_masks.shape)

                # true_masks=true_masks.unsqueeze(1)
                true_masks=true_masks.to(torch.float64)
                true_masks=F.interpolate(true_masks,size=(masks_pred.shape[2:]),mode='bilinear', align_corners=True)
                true_masks = torch.squeeze(true_masks)


                true_masks = true_masks.to(dtype=torch.int64)
                # true_mask_one_hot = get_one_hot(true_masks, 4)
                # true_mask_one_hot = torch.squeeze(true_mask_one_hot)
                # print('true_mask_one_hot',true_mask_one_hot.shape)
                # true_mask_onehot = true_mask_one_hot.permute(0,3,1,2)
                #
                # true_mask_onehot = true_mask_onehot.squeeze(0)
                # true_mask_onehot = true_mask_onehot.to(dtype=torch.float64)
                # pred_mask = masks_pred.to(dtype=torch.float64)
                # print('计算dice前：')
                # print('true_mask_onehot,pred_mask:',true_mask_onehot.shape,pred_mask.shape)

                list2 = multi_dice_coeff_list(masks_pred, true_masks)
                list3[0] += list2[0]
                list3[1] += list2[1]
                list3[2] += list2[2]
                list3[3] += list2[3]


                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                print('mask_pred:',masks_pred.shape)
                print('true_mask:',true_masks.shape)



                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
            list3[0], list3[1], list3[2], list3[3] = list3[0] / n_val, list3[1] / n_val, list3[2] / n_val, list3[
                3] / n_val

            print('dice:', list3)


        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=80,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.1,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    # net = UNet(n_channels=3, n_classes=1, bilinear=True)
    num_classes=4
    batch_size=16
    image_height=778
    image_width=549


    criterion = nn.CrossEntropyLoss()
    pretrained_model = "D:/Tea/BiSeNet/BiSeNet/source/pytorch-model/resnet18_v1.pth"
    net=BiSeNet(out_planes=num_classes, is_training=True,
                    criterion=criterion,
                    pretrained_model=pretrained_model,
                    )
    logging.info(f'Network:\n'
                 f'\tinput channels is 3\n'
                 f'\t{net.out_planes} output channels (classes)\n'
                )

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  # batch_size=args.batchsize,
                  batch_size=2,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
