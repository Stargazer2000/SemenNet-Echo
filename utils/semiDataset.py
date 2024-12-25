from utils.transform import crop, hflip, normalize, resize, blur, cutout
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import math
import random
from torchvision import transforms

class SemiDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, mode,scale=1,mask_suffix='_mask', unlabeled_dir=None, pseudo_dir=None,pseudo_mask=None):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.unlabeled_dir=unlabeled_dir
        self.pseudo_dir=pseudo_dir
        self.scale = scale
        self.mode = mode
        self.mask_suffix = mask_suffix
        self.pseudo_mask=pseudo_mask
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        if mode == 'semi_train':
            self.labeled_ids=[splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
            with open(self.pseudo_dir, 'r') as f:
                self.unlabeled_ids = f.read().splitlines()
            self.ids=self.labeled_ids+self.unlabeled_ids
            # self.ids = \
            #     self.labeled_ids * math.ceil(len(self.unlabeled_ids) / len(self.labeled_ids)) + self.unlabeled_ids
        if mode == 'label':
            with open(self.pseudo_dir, 'r') as f:
                self.ids = f.read().splitlines()

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small'
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        # print("process the image")
        # 因为mask中的像素值是标记后的0,1,2,3，所以对mask中的像素值不需要做预处理，只需要修改大小，而对原图img中的像素值，需要进行除255的操作。img中的像素值最小为33.
        if img_trans.max() > 10:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        # mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        # mask_file = glob(self.masks_dir + idx + '_gt.png')
        if self.mode == 'label':
            img_file = glob(self.unlabeled_dir + idx + '.jpg')
            img=Image.open(img_file[0])
            img = self.preprocess(img, self.scale)
            return {'img': torch.from_numpy(img).type(torch.FloatTensor),
                    'id' : idx}

        if idx in self.labeled_ids:
            img_file = glob(self.imgs_dir + idx + '.jpg')
            mask_file = glob(self.masks_dir + idx + '_gt.png')
        else:
            img_file = glob(self.unlabeled_dir + idx + '.jpg')
            mask_file = glob(self.pseudo_mask+ idx + '.jpg')
        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'
        # 用strong argumentation处理unlabeled image,省去了对所有图片的一般argumentation,包括crop和resize的操作

        if idx in self.unlabeled_ids:
            if random.random() < 0.8:
                img = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img)
            img = transforms.RandomGrayscale(p=0.2)(img)
            img = blur(img, p=0.5)
            # img, mask = cutout(img, mask, p=0.5)

        # img, mask = normalize(img, mask)


        img = self.preprocess(img, self.scale)
        mask = self.preprocess(mask, self.scale)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(SemiDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')
