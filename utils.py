import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np


def save_model(model, name,relu):
    if relu:
        torch.save(model.state_dict(), f'models/{name}_ReLU.pth')
    else:
        torch.save(model.state_dict(), f'models/{name}_noReLU.pth')


def load_model(path,model_type):
    model = model_type()
    model.load_state_dict(torch.load(f"models/{path}"))
    model.eval()
    return model


def load_batch(path):
    with open(path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        
    images = batch[b'data']      # (10000, 3072)
    labels = batch[b'labels']    # length 10000
    
    images = images.reshape(10000, 3, 32, 32)
    
    return images, np.array(labels)



class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.relu = False
        self.conv1r = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        init.kaiming_normal_(self.conv1r.weight)
        self.conv1g = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv1g.weight.data.fill_(0)
        self.conv1g.bias.data.fill_(1)
        self.conv1b = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv1b.weight.data.fill_(0)
        self.conv1b.bias.data.fill_(0)      
  
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2r = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        init.kaiming_normal_(self.conv2r.weight)
        self.conv2g = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2g.weight.data.fill_(0)
        self.conv2g.bias.data.fill_(1)
        self.conv2b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2b.weight.data.fill_(0)
        self.conv2b.bias.data.fill_(0)      
  
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        if self.relu:
            out = F.relu(self.bn1(self.conv1r(x)*self.conv1g(x)+self.conv1b(x.pow(2))))
            out = self.bn2(self.conv2r(out)*self.conv2g(out)+self.conv2b(out.pow(2)))
            out += self.shortcut(x)
            out = F.relu(out)
        else:
            out = self.bn1(self.conv1r(x)*self.conv1g(x)+self.conv1b(x.pow(2)))
            out = self.bn2(self.conv2r(out)*self.conv2g(out)+self.conv2b(out.pow(2)))
            out += self.shortcut(x)
            out = out
        return out
    


class QResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(QResNet, self).__init__()
        self.in_planes = 16
        self.input_channels = 3

        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)



    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        if relu:
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        else:
            out = self.bn1(self.conv1(x))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = F.avg_pool2d(out, out.size()[3])
            out = out.view(out.size(0), -1)
            out = self.linear(out)
        return out


def q_custom():
    return QResNet(BasicBlock, [3,3,3])

def qresnet20():
    return QResNet(BasicBlock, [3, 3, 3])


def qresnet32():
    return QResNet(BasicBlock, [5, 5, 5])


def qresnet44():
    return QResNet(BasicBlock, [7, 7, 7])


def qresnet56():
    return QResNet(BasicBlock, [9, 9, 9])


def qresnet110():
    return QResNet(BasicBlock, [18, 18, 18])


def qresnet1202():
    return QResNet(BasicBlock, [200, 200, 200])


