import codecs
import os

import numpy as np
import torch
from torch import distributions as dist
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from PIL import Image
import re


class SSL_Dataset(data.Dataset):
    def __init__(
        self,
        data,
        labels,
        mode,
        transform=None,
        **kwargs
    ):
        super().__init__()
        self.data = data
        self.targets = labels
        self.transform = transform
        self.mode = mode
   
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode in ['labeled_train']: 
            impath, target = self.data[index], self.targets[index]
            img = Image.open('../kaggle_data'+ impath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            target = target.long()
            return img, target
        elif self.mode in ['test', 'unlabeled_train']: 
            impath = self.data[index]
            img = Image.open('../kaggle_data'+ impath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            return img
        #################### EDIT HERE ####################
        ###################################################

