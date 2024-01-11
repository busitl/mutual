import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from itertools import product
from collections import namedtuple
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import time
from mutualLearning.model import Model1
from mutualLearning import utils


class Net(nn.Module):
    def __init__(self, num_class):
        super(Net, self).__init__()

        # encoder
        self.f = Model1().f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


fileName = 'results/linear_model.pth'


test_data = CIFAR10(root='data', train=True,
                    transform=utils.test_transform, download=True)
test_data = torch.utils.data.Subset(test_data, range(40000, 50000))
test_loader = DataLoader(test_data, batch_size=256,
                         shuffle=False, num_workers=4)


model = Net(num_class=10)
model.load_state_dict(torch.load(fileName))
model.cuda()
model.eval()

correct_num = 0
max_acc = 0
with torch.no_grad():
    start_time = time.time()
    total = 0
    for i in range(10):
        correct_num = 0
        for batch in test_loader:
            images = batch[0].cuda()
            labels = batch[1].cuda()
            preds = model(images)
            correct = preds.argmax(dim=1).eq(labels).sum().item()
            correct_num += correct
        accuracy_pred = correct_num / len(test_loader.dataset)
        print(accuracy_pred)
        if accuracy_pred > max_acc:
            max_acc = accuracy_pred
during = time.time()-start_time
print(fileName, max_acc, during)
