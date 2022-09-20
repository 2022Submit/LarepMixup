import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision


# Target Model definition
class MNIST_target_net(nn.Module):
    def __init__(self):
        super(MNIST_target_net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)

        self.fc1 = nn.Linear(64*4*4, 200)
        self.fc2 = nn.Linear(200, 200)
        self.logits = nn.Linear(200, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64*4*4)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = self.logits(x)
        return x



class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class LeNet(nn.Module):
    def __init__(self,n_channels=1,n_outputs=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(n_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        #x= F.softmax(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, name="ResNet18", n_channels=1, n_outputs=10):
        super().__init__()
        self.name = name
        self.n_outputs = n_outputs

        self.model = torchvision.models.resnet18(pretrained = False)
        self.model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_outputs)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

class ResNet34(nn.Module):
    def __init__(self, name="ResNet34", n_channels=1, n_outputs=10):
        super().__init__()
        self.name = name
        self.n_outputs = n_outputs

        self.model = torchvision.models.resnet34(pretrained = False)
        self.model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, n_outputs)

    def forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

class VGG16(nn.Module):
    def __init__(self, name="VGG16", n_channels = 1, n_outputs = 10):
        super(VGG16, self).__init__()
        self.name = name

        self.n_channels = n_channels
        self.features = self._make_layers([
            64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
            512, 512, 512, 'M', 512, 512, 512, 'M'
        ])
        self.classifier = nn.Linear(512, n_outputs)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.dropout(out, p = .5, training = self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, configuration):
        layers = []
        in_channels = self.n_channels
        for x in configuration:
            if x == 'M':
                layers += [ nn.MaxPool2d(kernel_size = 2, stride = 2) ]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size = 1, stride = 1)]
        return nn.Sequential(*layers)

class VGG13(nn.Module):
    def __init__(self, name="VGG13", n_channels = 1, n_outputs = 10):
        super(VGG13, self).__init__()
        self.name = name

        self.n_channels = n_channels
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Linear(512, n_outputs)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.dropout(out, p = .5, training = self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, configuration):
        layers = []
        in_channels = self.n_channels
        for x in configuration:
            if x == 'M':
                layers += [ nn.MaxPool2d(kernel_size = 2, stride = 2) ]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size = 1, stride = 1)]
        return nn.Sequential(*layers)

class VGG19(nn.Module):
    def __init__(self, name="VGG19", n_channels = 1, n_outputs = 10):
        super(VGG19, self).__init__()
        self.name = name

        self.n_channels = n_channels
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'])
        self.classifier = nn.Linear(512, n_outputs)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = torch.nn.functional.dropout(out, p = .5, training = self.training)
        out = self.classifier(out)
        return out

    def _make_layers(self, configuration):
        layers = []
        in_channels = self.n_channels
        for x in configuration:
            if x == 'M':
                layers += [ nn.MaxPool2d(kernel_size = 2, stride = 2) ]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size = 3, padding = 1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size = 1, stride = 1)]
        return nn.Sequential(*layers)

class Alexnet(nn.Module):
    def __init__(self, name='alexnet',n_channels=3, n_outputs=10):
        super(Alexnet, self).__init__()

        self.name = name
        self.num_classes = n_outputs

        self.conv1 = nn.Conv2d(n_channels, 48, 5, stride=1, padding=2)
        self.conv1.bias.data.normal_(0, 0.01)
        self.conv1.bias.data.fill_(0) 
        
        self.relu = nn.ReLU()        
        self.lrn = nn.LocalResponseNorm(2)        
        self.pad = nn.MaxPool2d(3, stride=2)
        
        self.batch_norm1 = nn.BatchNorm2d(48, eps=0.001)
        
        self.conv2 = nn.Conv2d(48, 128, 5, stride=1, padding=2)
        self.conv2.bias.data.normal_(0, 0.01)
        self.conv2.bias.data.fill_(1.0)  
        
        self.batch_norm2 = nn.BatchNorm2d(128, eps=0.001)
        
        self.conv3 = nn.Conv2d(128, 192, 3, stride=1, padding=1)
        self.conv3.bias.data.normal_(0, 0.01)
        self.conv3.bias.data.fill_(0)  
        
        self.batch_norm3 = nn.BatchNorm2d(192, eps=0.001)
        
        self.conv4 = nn.Conv2d(192, 192, 3, stride=1, padding=1)
        self.conv4.bias.data.normal_(0, 0.01)
        self.conv4.bias.data.fill_(1.0)  
        
        self.batch_norm4 = nn.BatchNorm2d(192, eps=0.001)
        
        self.conv5 = nn.Conv2d(192, 128, 3, stride=1, padding=1)
        self.conv5.bias.data.normal_(0, 0.01)
        self.conv5.bias.data.fill_(1.0)  
        
        self.batch_norm5 = nn.BatchNorm2d(128, eps=0.001)
        
        self.fc1 = nn.Linear(1152,512)
        self.fc1.bias.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0) 
        
        self.drop = nn.Dropout(p=0.5)
        
        self.batch_norm6 = nn.BatchNorm1d(512, eps=0.001)
        
        self.fc2 = nn.Linear(512,256)
        self.fc2.bias.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0) 
        
        self.batch_norm7 = nn.BatchNorm1d(256, eps=0.001)
        
        self.fc3 = nn.Linear(256,self.num_classes)
        self.fc3.bias.data.normal_(0, 0.01)
        self.fc3.bias.data.fill_(0) 
        
        self.soft = nn.Softmax()
        
    def forward(self, x):
        layer1 = self.batch_norm1(self.pad(self.lrn(self.relu(self.conv1(x)))))
        layer2 = self.batch_norm2(self.pad(self.lrn(self.relu(self.conv2(layer1)))))
        layer3 = self.batch_norm3(self.relu(self.conv3(layer2)))
        layer4 = self.batch_norm4(self.relu(self.conv4(layer3)))
        layer5 = self.batch_norm5(self.pad(self.relu(self.conv5(layer4))))
        flatten = layer5.view(-1, 128*3*3)
        fully1 = self.relu(self.fc1(flatten))
        fully1 = self.batch_norm6(self.drop(fully1))
        fully2 = self.relu(self.fc2(fully1))
        fully2 = self.batch_norm7(self.drop(fully2))
        logits = self.fc3(fully2)

        return logits

class HalfAlexnet(nn.Module):
    def __init__(self, name='half-alexnet',n_channels=3, n_outputs=10):
        super(HalfAlexnet, self).__init__()

        self.name = name
        self.num_classes = n_outputs

        self.conv1 = nn.Conv2d(n_channels, 24, 5, stride=1, padding=2)
        self.conv1.bias.data.normal_(0, 0.01)
        self.conv1.bias.data.fill_(0) 
        
        self.relu = nn.ReLU()        
        self.lrn = nn.LocalResponseNorm(2)        
        self.pad = nn.MaxPool2d(3, stride=2)
        
        self.batch_norm1 = nn.BatchNorm2d(24, eps=0.001)
        
        self.conv2 = nn.Conv2d(24, 64, 5, stride=1, padding=2)
        self.conv2.bias.data.normal_(0, 0.01)
        self.conv2.bias.data.fill_(1.0)  
        
        self.batch_norm2 = nn.BatchNorm2d(64, eps=0.001)
        
        self.conv3 = nn.Conv2d(64, 96, 3, stride=1, padding=1)
        self.conv3.bias.data.normal_(0, 0.01)
        self.conv3.bias.data.fill_(0)  
        
        self.batch_norm3 = nn.BatchNorm2d(96, eps=0.001)
        
        self.conv4 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.conv4.bias.data.normal_(0, 0.01)
        self.conv4.bias.data.fill_(1.0)  
        
        self.batch_norm4 = nn.BatchNorm2d(96, eps=0.001)
        
        self.conv5 = nn.Conv2d(96, 64, 3, stride=1, padding=1)
        self.conv5.bias.data.normal_(0, 0.01)
        self.conv5.bias.data.fill_(1.0)  
        
        self.batch_norm5 = nn.BatchNorm2d(64, eps=0.001)
        
        self.fc1 = nn.Linear(576,256)
        self.fc1.bias.data.normal_(0, 0.01)
        self.fc1.bias.data.fill_(0) 
        
        self.drop = nn.Dropout(p=0.5)
        
        self.batch_norm6 = nn.BatchNorm1d(256, eps=0.001)
        
        self.fc2 = nn.Linear(256,128)
        self.fc2.bias.data.normal_(0, 0.01)
        self.fc2.bias.data.fill_(0) 
        
        self.batch_norm7 = nn.BatchNorm1d(128, eps=0.001)
        
        self.fc3 = nn.Linear(128,10)
        self.fc3.bias.data.normal_(0, 0.01)
        self.fc3.bias.data.fill_(0) 
        
        self.soft = nn.Softmax()
        
    def forward(self, x):
        layer1 = self.batch_norm1(self.pad(self.lrn(self.relu(self.conv1(x)))))
        layer2 = self.batch_norm2(self.pad(self.lrn(self.relu(self.conv2(layer1)))))
        layer3 = self.batch_norm3(self.relu(self.conv3(layer2)))
        layer4 = self.batch_norm4(self.relu(self.conv4(layer3)))
        layer5 = self.batch_norm5(self.pad(self.relu(self.conv5(layer4))))
        flatten = layer5.view(-1, 64*3*3)
        fully1 = self.relu(self.fc1(flatten))
        fully1 = self.batch_norm6(self.drop(fully1))
        fully2 = self.relu(self.fc2(fully1))
        fully2 = self.batch_norm7(self.drop(fully2))
        logits = self.fc3(fully2)
        return logits



def CIFAR10():
    return VGG16(n_channels = 3)

def GTSRB():
    return VGG16(n_channels=3, n_outputs=43)

def MNIST():
    return LeNet()

def FashionMNIST():
    return LeNet()