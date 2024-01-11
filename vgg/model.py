import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg11
from torchvision.models.resnet import resnet50


class VGG(nn.Module):

    def __init__(self, init_weights=True):
        super(VGG, self).__init__()
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 40),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(40, 10),
        )
        self.g = nn.Sequential(
            nn.Linear(6272, 100, bias=False),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 4, bias=True)
        )

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.classifier(x)
        out = self.g(x)
        return F.normalize(out, dim=-1)


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(6272, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(2048, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            #nn.Dropout(),
            nn.Linear(512, 10)
        )

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class VGG2(nn.Module):

    def __init__(self, init_weights=True):
        super(VGG2, self).__init__()
        self.features = self.make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 40),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(40, 10),
        )
        self.g = nn.Sequential(
            nn.Linear(6272, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 128, bias=True)
        )

    def make_layers(slef, batch_norm=False):
        cfg = [16, 'M', 32, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128, 'M']
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d,
                               nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.classifier(x)
        out = self.g(x)
        return F.normalize(x, dim=-1), F.normalize(out, dim=-1)


class Model(nn.Module):
    def __init__(self, feature_dim=128):
        super(Model, self).__init__()

        self.f = []
        for name, module in resnet50().named_children():
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3,
                                   stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # encoder
        self.f = nn.Sequential(*self.f)
        # projection head
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feature_dim, bias=True)
        )

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)

class Res(nn.Module):
    def __init__(self, num_class=10):
        super(Res, self).__init__()

        # encoder
        self.f = Model().f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out

