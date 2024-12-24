
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(Block, self).__init__()
        self.n1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(out)))
        out += shortcut
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride):
        super(Bottleneck, self).__init__()
        self.n1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.n2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(out)))
        out = self.conv3(F.relu(self.n3(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, target_size):
        super().__init__()
        self.in_planes = hidden_size[0]
        self.conv1 = nn.Conv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2)
        self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion)
        self.fc = nn.Linear(hidden_size[3] * block.expansion, target_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def f(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(self.n4(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
    def get_feature(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.relu(self.n4(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return x

    def get_layer_parameters(self,idx):
        ls=[]
        if idx==1:
            conv1_weights = self.conv1.weight.data  # 获取权重
            conv1_weights_flattened = conv1_weights.view(-1)  # 展平为一维向量
            ls.append(conv1_weights_flattened)
            return ls
        elif idx==2:
            for module in self.layer1.modules():
                if isinstance(module, nn.Conv2d):  # 如果该模块是卷积层
                    ls.append(module.weight.data.view(-1))  # 展平并存储权重
            return ls
        elif idx==3:
            for module in self.layer2.modules():
                if isinstance(module, nn.Conv2d):  # 如果该模块是卷积层
                    ls.append(module.weight.data.view(-1))  # 展平并存储权重
            return ls
        elif idx==4:
            for module in self.layer3.modules():
                if isinstance(module, nn.Conv2d):  # 如果该模块是卷积层
                    ls.append(module.weight.data.view(-1))  # 展平并存储权重
            return ls
        elif idx==5:
            for module in self.layer4.modules():
                if isinstance(module, nn.Conv2d):  # 如果该模块是卷积层
                    ls.append(module.weight.data.view(-1))  # 展平并存储权重
            return ls
        elif idx==6:
            # 获取fc层的权重和偏置
            fc_weight = self.fc.weight.data  # 获取fc层的权重
            fc_bias = self.fc.bias.data  # 获取fc层的偏置

            # 展平权重和偏置为一维
            fc_weight_flattened = fc_weight.view(-1)
            fc_bias_flattened = fc_bias.view(-1)

            # 拼接权重和偏置
            fc_params = torch.cat((fc_weight_flattened, fc_bias_flattened))
            ls.append(fc_params)
            return ls
        else:
            return '索引超出范围'

    def get_feature_layer(self,x,idx):
        x = self.conv1(x)
        if idx==1:
            return x
        x = self.layer1(x)
        if idx==2:
            return x
        x = self.layer2(x)
        if idx==3:
            return x
        x = self.layer3(x)
        if idx==4:
            return x
        x = self.layer4(x)
        if idx==5:
            return x
        x = F.relu(self.n4(x))
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        if idx==6:
            return x
        return '索引超出范围'  
        
    def forward(self, input):
        return self.f(input)

def resnet9(data_shape=[3,32,32], target_size=10, ):
    model = ResNet(data_shape, [64, 128, 256, 512], Block, [1, 1, 1, 1], target_size)
    return model


def resnet18(data_shape=[3,32,32], target_size=10, ):
    model = ResNet(data_shape, [64, 128, 256, 512], Block, [2, 2, 2, 2], target_size)
    return model