import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision import transforms
from tqdm import tqdm
import utils
import torch.nn.functional as F
from torch.autograd import Variable
from model import Net
import time
from EarlyStopping import EarlyStopping
import os

batch_size = 512
device = utils.get_device()
epochs = 400


def mkdir_if_not_exist(path):
    # 若目录path不存在，则创建目录
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


def train_val(net, data_loader, train_optimizer, T, a):
    is_train = train_optimizer is not None
    model_num = len(net)
    total_loss = [0.0, 0.0]
    total_correct = [0.0, 0.0]
    loss = [0.0, 0.0]
    for i in range(model_num):
        net[i].train() if is_train else net[i].eval()
    total_num, data_bar = 0.0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            outputs = []
            for model in net:
                outputs.append(model(data))

            for i in range(model_num):
                ce_loss = F.cross_entropy(outputs[i], target)  # 当前模型的交叉熵损失值
                kl_loss = 0
                for j in range(model_num):  # KL散度 重点
                    if i != j:
                        kl_loss += F.kl_div(F.log_softmax(outputs[i]/T, dim=1),
                                            F.softmax(Variable(outputs[j])/T, dim=1))*T*T
                loss = a*ce_loss + (1-a) * kl_loss / (model_num - 1)
                if is_train:
                    train_optimizer[i].zero_grad()
                    loss.backward()
                    train_optimizer[i].step()

                total_loss[i] += loss.item() * data.size(0)
                prediction = torch.argsort(outputs[i], dim=-1, descending=True)
                total_correct[i] += torch.sum(
                    (prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_num += data.size(0)
            data_bar.set_description('{} Epoch: [{}/{}] Loss1: {:.4f} ACC1: {:.2f}%,Loss2: {:.4f} ACC2: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss[0] / total_num,
                                             total_correct[0] / total_num * 100, total_loss[1] / total_num, total_correct[1] / total_num * 100))
    return total_loss[0] / total_num, total_correct[0] / total_num * 100, total_loss[1] / total_num, total_correct[1] / total_num * 100


model_path1, model_path2 = 'vgg/results/rotation_linear.pth', 'vgg/results/contrast_linear.pth'

train_data = CIFAR10(root='data', train=True,
                     transform=utils.train_transform, download=True)
train_data = torch.utils.data.Subset(train_data, range(3000))
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=16)
test_data = CIFAR10(root='data', train=False,
                    transform=utils.test_transform, download=True)
test_loader = DataLoader(test_data, batch_size=batch_size,
                         shuffle=False, num_workers=4)


for T in [25, 30]:
    for a in [0.95]:
        model1 = Net()
        model1.load_state_dict(torch.load(model_path1))
        model2 = Net()
        model2.load_state_dict(torch.load(model_path2))

        optimizer1 = optim.Adam(model1.parameters(), lr=1e-4)
        optimizer2 = optim.Adam(model2.parameters(), lr=1e-4)
        optimizer = [optimizer1, optimizer2]

        net1 = nn.DataParallel(model1)
        net1.to(device)
        net2 = nn.DataParallel(model2)
        net2.to(device)
        net = [net1, net2]

        results = {'train_loss1': [], 'train_acc1': [],
                   'test_loss1': [], 'test_acc1': [], 'train_loss2': [], 'train_acc2': [],
                   'test_loss2': [], 'test_acc2': []}

        best_acc1 = 0.0
        best_acc2 = 0.0
        earlyStop1 = EarlyStopping()
        earlyStop2 = EarlyStopping()
        flag1 = flag2 = False
        path = 'T='+str(T)+'_a='+str(a)
        mkdir_if_not_exist(['vgg', 'mutual_result', path])
        storage = ['vgg', 'mutual_result', path, 'result.csv']
        storage1 = ['vgg', 'mutual_result', path, 'rotation.pth']
        storage2 = ['vgg', 'mutual_result', path, 'contrast.pth']
        path = os.path.join(*storage)
        path1 = os.path.join(*storage1)
        path2 = os.path.join(*storage2)
        for epoch in range(1, epochs + 1):
            train_loss1, train_acc1, train_loss2, train_acc2 = train_val(
                net, train_loader, optimizer, T, a)
            results['train_loss1'].append(train_loss1)
            results['train_acc1'].append(train_acc1)
            results['train_loss2'].append(train_loss2)
            results['train_acc2'].append(train_acc2)
            if epoch % 5 == 0:
                test_loss1, test_acc1, test_loss2, test_acc2 = train_val(
                    net, test_loader, None, T, a)
                results['test_loss1'].append(test_loss1)
                results['test_acc1'].append(test_acc1)
                results['test_loss2'].append(test_loss2)
                results['test_acc2'].append(test_acc2)
                if test_acc1 > best_acc1:
                    best_acc1 = test_acc1
                    torch.save(model1.state_dict(),
                               path1)
                    print("save model1")
                if test_acc2 > best_acc2:
                    best_acc2 = test_acc2
                    torch.save(model1.state_dict(),
                               path2)
                    print("save model2")

            else:
                results['test_loss1'].append(0)
                results['test_acc1'].append(0)
                results['test_loss2'].append(0)
                results['test_acc2'].append(0)

            data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
            data_frame.to_csv(path, index_label='epoch')
            if flag1 == False:
                earlyStop1(train_loss1)
            if flag2 == False:
                earlyStop2(train_loss2)
            if earlyStop1.early_stop:
                flag1 = True
            if earlyStop2.early_stop:
                flag2 = True
            if flag1 == True and flag2 == True:
                print('early stop')
                break
