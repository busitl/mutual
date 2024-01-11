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


batch_size = 128
device = utils.get_device()
epochs = 200


def train_val(net, data_loader, train_optimizer):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(
        data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target in data_bar:
            data, target = data.to(device), target.to(device)
            out = net(data)
            loss = loss_criterion(out, target)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)
            total_correct_1 += torch.sum(
                (prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum(
                (prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100


model_path = './results/vgg11_rotation_model.pth'
train_data = CIFAR10(root='data', train=True,
                     transform=utils.train_transform, download=True)
train_data = torch.utils.data.Subset(train_data, range(3000))
train_loader = DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=16)
test_data = CIFAR10(root='data', train=False,
                    transform=utils.test_transform, download=True)
test_loader = DataLoader(test_data, batch_size=batch_size,
                         shuffle=False, num_workers=4)

tea = VGG()

model = Net()
tea.load_state_dict(torch.load(model_path))

model.features = tea.features
model.avgpool = tea.avgpool

indx = 0
for i in model.features:
    #print(i)
    indx = indx+1

#print(indx)

for i in range(0, indx):
    #print(model.features[i])
    for param in model.features[i].parameters():
        #print(param)
        param.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=1e-2)


net = nn.DataParallel(model)
net.to(device)

loss_criterion = nn.CrossEntropyLoss()
results = {'train_loss': [], 'train_acc@1': [], 'train_acc@5': [],
           'test_loss': [], 'test_acc@1': [], 'test_acc@5': []}

best_acc = 0.0
for epoch in range(1, epochs + 1):
    train_loss, train_acc_1, train_acc_5 = train_val(
        net, train_loader, optimizer)
    results['train_loss'].append(train_loss)
    results['train_acc@1'].append(train_acc_1)
    results['train_acc@5'].append(train_acc_5)
    test_loss, test_acc_1, test_acc_5 = train_val(net, test_loader, None)
    results['test_loss'].append(test_loss)
    results['test_acc@1'].append(test_acc_1)
    results['test_acc@5'].append(test_acc_5)

    data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
    data_frame.to_csv('./results/rotation_linear1.csv', index_label='epoch')
    if test_acc_1 > best_acc:
        best_acc = test_acc_1
        torch.save(model.state_dict(), './results/rotation_linear1.pth')
        print("save it")
