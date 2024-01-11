import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import pandas as pd
import time
import json
from EarlyStopping import EarlyStopping
import utils
from utils import train_transform, CIFAR10Pair, get_device
from tqdm import tqdm
from rotationLearning import RotationDataset
from model import VGG
lr = 0.001
batch_size = 512
temperature = 0.5
epochs = 400

transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(),
    ])

train_set = torchvision.datasets.CIFAR10(
    root='data',
    train=True,
    download=True,
    transform=transform
)
train_data1 = RotationDataset(train_set)

val_set = torchvision.datasets.CIFAR10(
    root='data',
    train=False,
    download=True,
    transform=transform
)
train_data2 = RotationDataset(val_set)


def train(net, data_loader1, data_loader2, train_optimizer):
    net.train()
    total_loss, total_num, train_bar1, train_bar2 = 0.0, 0, tqdm(
        data_loader1), tqdm(data_loader2)
    total_top1, total_top5 = 0.0, 0.0
    for train_bar in [train_bar1, train_bar2]:
        for batch in train_bar:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            images = images.view(
                images.size()[0]*images.size()[1], images.size()[2], images.size()[3], images.size()[4]).squeeze()
            labels = labels.view(labels.size()[0]*labels.size()[1]).squeeze()
            outputs = net(images)
            loss = loss_criterion(outputs, labels)

            train_optimizer.zero_grad()
            loss.backward()
            train_optimizer.step()

            pred_labels = outputs.argsort(dim=-1, descending=True)
            total_top1 += torch.sum(
                (pred_labels[:, :1] == labels.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_top5 += torch.sum(
                (pred_labels[:, :5] == labels.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            total_num += batch_size*4
            total_loss += loss.item() * batch_size*4
            train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} Acc@1:{:.2f}% Acc@5:{:.2f}%'.format(
                epoch, epochs, total_loss / total_num, total_top1 / total_num * 100, total_top5 / total_num * 100))
    return total_top1 / total_num * 100, total_top5 / total_num * 100, total_loss / total_num


device = get_device()
model_path = './results/vgg11_rotation_model.pth'
model = VGG()
model.load_state_dict(torch.load(model_path))
Early_stop = EarlyStopping()
train_loader1 = DataLoader(train_data1, batch_size=batch_size,
                           shuffle=True, num_workers=16, drop_last=True)
train_loader2 = DataLoader(train_data2, batch_size=batch_size,
                           shuffle=True, num_workers=16, drop_last=True)

optimizer = optim.Adam(model.parameters(), lr=lr)
net = nn.DataParallel(model)
net.to(device)

results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
loss_criterion = nn.CrossEntropyLoss()
best_loss = 10000.0

for epoch in range(1, epochs + 1):
    test_acc_1, test_acc_5, train_loss = train(
        net, train_loader1, train_loader2, optimizer)
    results['train_loss'].append(train_loss)
    results['test_acc@1'].append(test_acc_1)
    results['test_acc@5'].append(test_acc_5)
    # save statistics
    data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    data_frame.to_csv(
        './results/vgg11_roration_statistics.csv', index_label='epoch')
    if train_loss < best_loss:
        best_loss = train_loss
        torch.save(model.state_dict(),
                   './results/vgg11_rotation_model.pth')
