import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm
import utils
from model import VGG, Net, VGG2
from torchvision.models.vgg import vgg11

from model import Res

batch_size = 128

net = Res()
net.load_state_dict(torch.load(
    'vgg/mutual_result/newT=10_a=0.9/rotation2.pth'))
test_data = CIFAR10(root='data', train=False,
                    transform=utils.test_transform, download=True)
test_loader = DataLoader(test_data, batch_size=batch_size,
                         shuffle=False, num_workers=4)


def eval():
    net.eval()
    correct = 0
    correct3 = 0
    correct5 = 0
    total = 0

    classnum = 10
    target_num = torch.zeros((1, classnum))
    predict_num = torch.zeros((1, classnum))
    acc_num = torch.zeros((1, classnum))

    for batch_idx, (inputs, targets) in enumerate(test_loader):
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).sum()
        pre_mask = torch.zeros(outputs.size()).scatter_(
            1, predicted.view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(outputs.size()).scatter_(
            1, targets.data.view(-1, 1), 1.)

        y = targets.data.view(-1, 1)

        maxk = 5
        _, pred5 = outputs.topk(maxk, 1, True, True)
        correct5 += torch.eq(pred5, y).sum().float().item()

        maxk = 3
        _, pred3 = outputs.topk(maxk, 1, True, True)
        correct3 += torch.eq(pred3, y).sum().float().item()

        target_num += tar_mask.sum(0)
        acc_mask = pre_mask*tar_mask
        acc_num += acc_mask.sum(0)

    recall = acc_num/target_num
    precision = acc_num/predict_num
    accuracy = acc_num.sum(1)/target_num.sum(1)
    acc3 = correct3/total
    acc5 = correct5/total

# 精度调整
    recall = (recall.cpu().numpy()[0]*100).round(3)
    precision = (precision.cpu().numpy()[0]*100).round(3)
    accuracy = (accuracy.cpu().numpy()[0]*100).round(3)
    acc3 = acc3*100
    acc5 = acc5*100

# 打印格式方便复制
    print('recall', " ".join('%s' % id for id in recall))
    print('precision', " ".join('%s' % id for id in precision))
    print('accuracy', accuracy)
    print('acc3', acc3)
    print('acc5', acc5)


eval()
