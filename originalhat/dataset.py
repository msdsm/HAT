import torch
import torchvision
import glob
from PIL import Image
import cv2
from torchvision.transforms.functional import normalize
import numpy as np

from basicsr.utils import img2tensor

class HAT_Dataset(torch.utils.data.Dataset):
    def __init__(self, lowimg_path, gtimg_path, lowimg_files, gtimg_files, transform):
        '''
        data_path : ディレクトリパス
        data_pathの下に"input", "groundTruth"ディレクトリがある
        それぞれ3つのディレクトリから取り出す処理を__getitem__に実装
        tuple of listを返す
        そうすると自動でdataloaderはlist of tupleになってくれるらしい
        (対応するものの名前を一致させておく必要はある)
        '''
        self.lowimg_path = lowimg_path
        self.gtimg_path = gtimg_path
        self.lowimg_files = lowimg_files
        self.gtimg_files = gtimg_files
        self.transform = transform
    
    def __getitem__(self, index):
        # hat/data/imagenet_paired_dataset.py
        lowimg = Image.open(self.lowimg_path + self.lowimg_files[index])
        gtimg = Image.open(self.gtimg_path + self.gtimg_files[index])
        lowimg = self.transform(lowimg)
        gtimg = self.transform(gtimg)
        # lowimg.requires_grad = True
        # gtimg.requires_grad = True

        '''
        # ここからHAT/hat/data/imagenet_paired_dataset.py参考にした
        lowimg, gtimg = img2tensor([lowimg, gtimg], float32=True)
        if self.mean is not None or self.std is not None:
            normalize(lowimg, self.mean, self.std, inplace=True)
            normalize(gtimg, self.mean, self.std, inplace=True)

        上のtransformでもうやってることではないか？？
        '''
        return (lowimg, gtimg)
    
    def __len__(self):
        return len(self.lowimg_files)