# encoding=utf-8
# created by yan-x-p 2019.04.17
from torch.utils.data import *
import os
from PIL import Image

class CarDataLoader(Dataset):
    def __init__(self, img_dir,label_path,chars,transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        self.labels = []
        f = open(label_path,'r')
        for line in f.readlines():
            label, img = line.strip().split(',  ')
            self.labels.append(label)
            self.img_paths.append(os.path.join(img_dir,img))
        self.transform = transform
        self.mapping = chars
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img_label = self.labels[index]
        label  = []
        for i in img_label:
            label.append(self.mapping[i])
        img = Image.open(img_name).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img,label

class CarTestDataLoader(Dataset):
    def __init__(self, img_dir,label_path,transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        f = open(label_path,'r')
        for line in f.readlines():
            label, img = line.strip().split(',  ')
            self.img_paths.append(os.path.join(img_dir,img))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = Image.open(img_name).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img,img_name.split('/')[-1]