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
from model import VGG2
import utils
from utils2 import train_transform, CIFAR10Pair, get_device
from tqdm import tqdm

lr = 0.001
batch_size = 256
temperature = 0.5
epochs = 200

train_data = utils.CIFAR10Pair(
    root='data', train=True, transform=train_transform, download=True)


def train(net, data_loader1, train_optimizer):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader1)
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2 = pos_1.to(device), pos_2.to(device)
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)
        out = torch.cat([out_1, out_2], dim=0)
        sim_matrix = torch.exp(
            torch.mm(out, out.t().contiguous()) / temperature)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
        sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
        pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()
        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description(
            'Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
    return total_loss / total_num


device = get_device()
model = VGG2()
Early_stop = EarlyStopping()
train_loader = DataLoader(train_data, batch_size=batch_size,
                          shuffle=True, num_workers=4, drop_last=True)
optimizer = optim.Adam(model.parameters(), lr=lr)
net = nn.DataParallel(model)
net.to(device)

results = {'train_loss': []}
loss_my = 10000
for epoch in range(1, epochs + 1):
    train_loss = train(net, train_loader, optimizer)
    results['train_loss'].append(train_loss)
    data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    data_frame.to_csv('vgg/results/contrast1.csv', index_label='epoch')
    if loss_my > train_loss:
        loss_my = train_loss
        torch.save(model.state_dict(), 'vgg/results/contrast.pth')
        print("save it")
