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
from BiSeNet.networkAttn import BiSeNet
from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

from BiSeNet.loss_opr import SigmoidFocalLoss, ProbOhemCrossEntropy2d

dir_img='D:/InstrumentProject/CAMUS/training/pictures_8/'
dir_mask = 'D:/InstrumentProject/CAMUS/training/masks_6/'
# dir_img='/data/pictures_9/'
# dir_mask='/data/masks_6/'
dir_checkpoint = 'FinalCheckpoints/'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    # 以下是PLANet的训练
    # optimizer=optim.SGD(net.parameters(), lr=lr, weight_decay=1e-4, momentum=0.9)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()


        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                # assert imgs.shape[1] == net.n_channels, \
                #     f'Network has been defined with {net.n_channels} input channels, ' \
                #     f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                #     'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)

                optimizer.zero_grad()

                masks_pred = net(imgs)




                # 使用F.interpolate方法会带来椒盐噪声,对左心房和左心室的分割效果影像尤其明显
                # true_masks=true_masks.unsqueeze(1)
                # true_masks=F.interpolate(true_masks,size=(masks_pred.shape[2:]),mode='bilinear', align_corners=True)
                # print('padding之前true_masks和masks_pred的shape',true_masks.shape,masks_pred.shape)
                diffY = masks_pred.size()[2] - true_masks.size()[2]
                diffX = masks_pred.size()[3] - true_masks.size()[3]

                true_masks = F.pad(true_masks, [diffX // 2, diffX - diffX // 2,
                                              diffY // 2, diffY - diffY // 2])





                true_masks = torch.squeeze(true_masks)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)
                loss = criterion(masks_pred, true_masks)


                epoch_loss +=loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    for tag, value in net.named_parameters():
                        tag = tag.replace('.', '/')
                        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                        # writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, val_loader, device,epoch)
                    scheduler.step(val_score['mean'])
                    writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        list = val_score['list']
                        # logging.info('Validation cross entropy: {}'.format(val_score))
                        logging.info('dice0: {},dice1:{},dice2:{},dice3:{}'.format(list[0], list[1], list[2], list[3]))
                        logging.info('mean dice:{}'.format(val_score['mean']))
                        writer.add_scalar('Loss/test', val_score['mean'], global_step)
                    else:
                        logging.info('Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score['mean'], global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', true_masks, global_step)
                        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)


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
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=700,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.2,
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
    image_height=512
    image_width=512


    criterion = nn.CrossEntropyLoss()
    pretrained_model = "D:/Tea3/BiSeNet_3/BiSeNet/source/pytorch-model/resnet18_v1.pth"
    # pretrained_model="/tmp/pycharm_project_394/source/pytorch-model/resnet18_v1.pth"
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
                  batch_size=24,
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
