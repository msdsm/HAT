import torch
import torchvision
import glob
from PIL import Image

class HAT_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, lowimg_files, maskimg_files, gtimg_files, transform):
        '''
        data_path : ディレクトリパス
        data_pathの下に"input", "mask", "groundTruth"ディレクトリがある
        それぞれ3つのディレクトリから取り出す処理を__getitem__に実装
        tuple of listを返す
        そうすると自動でdataloaderはlist of tupleになってくれるらしい
        (対応するものの名前を一致させておく必要はある)
        '''
        self.data_path = data_path
        self.lowimg_files = lowimg_files
        self.mskimg_files = maskimg_files
        self.gtimg_files = gtimg_files
        self.transform = transform
    
    def __getitem__(self, index):
        # hat/data/imagenet_paired_dataset.py
        lowimg = Image.open(self.data_path + self.lowimg_files[index])
        maskimg = Image.open(self.data_path + self.maskimg_files[index])
        gtimg = Image.open(self.data_path + self.gtimg_files[index])
        lowimg = self.transform(lowimg)
        maskimg = self.transform(maskimg)
        gtimg = self.transform(gtimg)
        return (lowimg, maskimg, gtimg)
    
    def __len__(self):
        return len(self.lowimg_files)