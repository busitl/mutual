import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision import datasets, models
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import random
import numpy as np
import torch.utils.data as data
from PIL import Image


class RotationDataset(Dataset):

    def __init__(self, dataset):

        # Output of pretransform should be PIL images
        self.postTransform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms. Normalize(mean=[0.4914, 0.4822, 0.4465],
                                      std=[0.2023, 0.1994, 0.2010]),

            ])
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img0, clss = self.dataset[idx]

        img1 = img0.rotate(90)
        img2 = img0.rotate(180)
        img3 = img0.rotate(270)

        img_list = [img0, img1, img2, img3]

        arr4 = np.arange(4)
        np.random.shuffle(arr4)
        img_newList = [img_list[arr4[0]], img_list[arr4[1]],
                       img_list[arr4[2]], img_list[arr4[3]]]

        if self.postTransform:
            sampleList = list(
                map(lambda pilim: self.postTransform(pilim), img_newList))
        else:
            sampleList = list(
                map(lambda pilim: transforms.ToTensor(pilim), img_newList))
        sample = torch.stack(sampleList)
        rotation_labels = torch.LongTensor(arr4)
        return sample, rotation_labels
