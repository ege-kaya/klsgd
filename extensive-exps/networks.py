import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

class Linear(nn.Module):
    def __init__(self, dx, num_classes):
        super(Linear, self).__init__()
        self.dx = dx
        self.fc = nn.Linear(self.dx, num_classes)
    def forward(self, x):
        out = self.fc(x.view(-1,self.dx))
        return out


class LeNet(nn.Module):
    def __init__(self, nc, nh, hw, num_classes):
        input_shape = (nc,nh,hw)
        super(LeNet, self).__init__()
        self.maxpool = nn.MaxPool2d((2,2))      
        self.conv1 = nn.Conv2d(nc,64,5)    
        self.conv2 = nn.Conv2d(64,64,5)        
        self.flat_shape = self.get_flat_shape(input_shape)            
        self.fc1 = nn.Linear(self.flat_shape, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def get_flat_shape(self, input_shape):
        dummy = Variable(torch.zeros(1, *input_shape))
        dummy = self.maxpool(self.conv1(dummy))
        dummy = self.maxpool(self.conv2(dummy))
        return dummy.data.view(1, -1).size(1)
        
    def forward(self, x_in):        
        # conv 1
        x = self.conv1(x_in)
        x = self.maxpool(x)
        x = F.relu(x)
        # conv 2
        x = self.conv2(x)
        x = self.maxpool(x)
        x = F.relu(x)
        # flatten
        x = x.view(-1,self.flat_shape)
        # fc 1
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        # fc 2
        x_out1 = self.fc2(x)    
        return x_out1      

    

class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, nc=3, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18(nc=3, num_classes=10):
    return PreActResNet(PreActBlock, [2,2,2,2], nc, num_classes)