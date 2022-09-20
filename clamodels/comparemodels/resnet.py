'''ResNet in PyTorch.

BasicBlock and Bottleneck module is from the original ResNet paper:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

PreActBlock and PreActBottleneck module is from the later paper:
[2] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Identity Mappings in Deep Residual Networks. arXiv:1603.05027
'''
from distutils.log import error
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

import random
import numpy as np
import utils.sampler


def hidden_patchmixup_process(out, y, defense_mode):      
    batch_size = out.size()[0]
    index = torch.randperm(batch_size).cuda()                           

    """
    index: tensor([0, 1, 2, 3], device='cuda:0')
    indices.len: 4
    out.shape: torch.Size([4, 64, 32, 32])
    y.shape: torch.Size([4, 10])
    """                      
    is_2d = True if len(out.size()) == 2 else False                 
    m = utils.sampler.BernoulliSampler(out.size(0), out.size(1), is_2d, p=None)                                

    lam = []
    for i in range(len(m)):
        lam_i = (torch.nonzero(m[i]).size(0)) / m.size(1)
        lam.append(lam_i)

    lam = np.asarray(lam)
    lam = torch.tensor(lam).unsqueeze(1)

    m1 = m.cpu()
    m2 = (1.-m).cpu()
    lam1 = lam.cpu()
    lam2 = (1.-lam).cpu() 

    """
    lam1: tensor([[0.5039],[0.4727],[0.5547],[0.4922]], dtype=torch.float64)
    lam2: tensor([[0.4961],[0.5273],[0.4453],[0.5078]], dtype=torch.float64)
    """

    """
    m1.shape: torch.Size([4, 64, 1, 1])
    m2.shape: torch.Size([4, 64, 1, 1])
    lam1.shape: torch.Size([4, 1])
    lam2.shape: torch.Size([4, 1])
    """

    w1 = out.cpu()
    w2 = out[index,:].cpu()        
    y1 = y.cpu()
    y2 = y[index,:].cpu()

    """
    y1: tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    y2: tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    """
    out = m1*w1 + m2*w2
    mixed_y = lam1*y1 + lam2*y2

    """
    out.shape: torch.Size([4, 128, 16, 16])
    mixed_y.shape: torch.Size([4, 10])
    mixed_y: tensor([
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.4688, 0.0000, 0.0000, 0.5312], 
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5781, 0.0000, 0.0000, 0.4219],
                    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000]
                    ], dtype=torch.float64)
    mixed_y: tensor([[0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
                    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]], dtype=torch.float64)                    
    """                               
    out=out.cuda()
    mixed_y=mixed_y.cuda()

    return out, mixed_y

def hidden_manifoldmixup_process(out, y, defense_mode, beta_alpha):
    alpha=beta_alpha
    lam = np.random.beta(alpha, alpha)                              
    batch_size = out.size()[0]
    index = torch.randperm(batch_size).cuda()                                                 

    """
    index: tensor([24, 31, ..., 12, 29, 19],  device='cuda:0')
    indices.len: 32
    lam: 0.09522716670648239
    out.shape: torch.Size([32, 128, 16, 16])
    y.shape: torch.Size([32, 10])
    """
    out = lam*out + (1-lam)*out[index,:]                        
    mixed_y = lam*y + (1-lam)*y[index,:]

    return out, mixed_y

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
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

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out)
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = conv3x3(3,64)
        self.bn1 = nn.BatchNorm2d(64)
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

    def forward(self, x, lin=0, lout=5, y=None, defense_mode=None, beta_alpha=None, imagenetmixed10=None):

        if defense_mode in ['manifoldmixup','patchmixup']:

            """
            defense_mode patchmixup
            y.shape torch.Size([4, 10])
            layer_mix: 2
            """
            layer_mix = random.randint(1, 3)                                          
        
        else:
            layer_mix = None

        out = x                                                                         

        if lin < 1 and lout > -1:
            out = self.conv1(out)
            out = self.bn1(out)
            out = F.relu(out)

        if lin < 2 and lout > 0:
            out = self.layer1(out)                                                                     

        if lin < 3 and lout > 1:
            if layer_mix == 1:                                                         
                if defense_mode == 'manifoldmixup':
                    out, mixed_y = hidden_manifoldmixup_process(out, y, defense_mode, beta_alpha)
                elif defense_mode == 'patchmixup':
                    out, mixed_y = hidden_patchmixup_process(out, y, defense_mode)
            out = self.layer2(out)                                                     

        if lin < 4 and lout > 2:
            if layer_mix == 2:                                                         
                if defense_mode == 'manifoldmixup':
                    out, mixed_y = hidden_manifoldmixup_process(out, y, defense_mode, beta_alpha)
                elif defense_mode == 'patchmixup':
                    out, mixed_y = hidden_patchmixup_process(out, y, defense_mode)     
            out = self.layer3(out)                                                      

        if lin < 5 and lout > 3:
            if layer_mix == 3:                                                         
                if defense_mode == 'manifoldmixup':
                    out, mixed_y = hidden_manifoldmixup_process(out, y, defense_mode, beta_alpha)
                elif defense_mode == 'patchmixup':
                    out, mixed_y = hidden_patchmixup_process(out, y, defense_mode)            
            out = self.layer4(out)                                                      

        if lout > 4:
            if imagenetmixed10 == True:
                avg = nn.AdaptiveAvgPool2d((1, 1))      
                out = avg(out)
            else:
                out = F.avg_pool2d(out, 4)    
            out = out.view(out.size(0), -1)
            out = self.linear(out)

        if defense_mode in ['manifoldmixup','patchmixup']:
            return out, mixed_y
        else:
            return out

def ResNet18():
    return ResNet(PreActBlock, [2,2,2,2])

def ResNet34():
    return ResNet(PreActBlock, [3,4,6,3])   

def ResNet50():
    return ResNet(PreActBlock, [3,4,6,3])   

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

def preactresnet18():
    return ResNet18()

def preactresnet34():
    return ResNet34()

def preactresnet50():
    return ResNet50()

