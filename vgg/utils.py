from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import torch

class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target


train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

def get_device():
    import platform
    if 'Windows' != platform.system():
        import os
        os.system(
            'nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
        gpu_memory = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
        os.system('rm tmp.txt')

    gpu_list = np.argsort(gpu_memory)[::-1]

    gpu_list_str = ','.join(map(str, gpu_list))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return device