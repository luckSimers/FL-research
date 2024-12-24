import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter



class DecConv2d(nn.Conv2d):
    def __init__(self, in_channels: int, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DecConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.sigma_weight = Parameter(copy.deepcopy(self.weight.data) / 2)
        self.phi_weight = Parameter(copy.deepcopy(self.weight.data) / 2)
        self.weight = None # type: ignore
        if self.bias is not None:
            self.sigma_bias = Parameter(copy.deepcopy(self.bias.data) / 2)
            self.phi_bias = Parameter(copy.deepcopy(self.bias.data) / 2)
            self.bias_ = torch.as_tensor(self.bias)
            self.bias = None
        else:
            self.register_parameter('bias_', None)

    def forward(self, input):
        if self.bias_ is not None:
            return self._conv_forward(input, self.sigma_weight + self.phi_weight, self.sigma_bias + self.phi_bias)
        else:
            return self._conv_forward(input, self.sigma_weight + self.phi_weight, self.bias_)
        

class DecLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(DecLinear, self).__init__(
            in_features, out_features, bias=bias)
        self.sigma_weight = Parameter(copy.deepcopy(self.weight.data) / 2)
        self.phi_weight = Parameter(copy.deepcopy(self.weight.data) / 2)
        self.weight = None # type: ignore
        if self.bias is not None:
            self.sigma_bias = Parameter(copy.deepcopy(self.bias.data) / 2)
            self.phi_bias = Parameter(copy.deepcopy(self.bias.data) / 2)
            self.bias_ = torch.as_tensor(self.bias)
            self.bias = None
        else:
            self.register_parameter('bias_', None)

    def forward(self, input):
        if self.bias_ is not None:
            return F.linear(input, self.sigma_weight + self.phi_weight, self.sigma_bias + self.phi_bias )
        else:
            return F.linear(input, self.sigma_weight + self.phi_weight, self.bias_)

class Block(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride):
        super(Block, self).__init__()
        self.n1 = nn.BatchNorm2d(in_planes)
        self.conv1 = DecConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.n2 = nn.BatchNorm2d(planes)
        self.conv2 = DecConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = DecConv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)

    def forward(self, x):
        out = F.relu(self.n1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.n2(out)))
        out += shortcut
        return out


class ResNet(nn.Module):
    def __init__(self, data_shape, hidden_size, block, num_blocks, target_size):
        super().__init__()
        self.in_planes = hidden_size[0]
        self.conv1 = DecConv2d(data_shape[0], hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3], stride=2)
        self.n4 = nn.BatchNorm2d(hidden_size[3] * block.expansion)
        self.fc = nn.Linear(hidden_size[3] * block.expansion, target_size)
        # self.fc = DecLinear(hidden_size[3] * block.expansion, target_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and isinstance(m, DecConv2d):
                nn.init.kaiming_normal_(m.sigma_weight, mode='fan_out', nonlinearity='relu')
                nn.init.kaiming_normal_(m.phi_weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear) and not isinstance(m, DecLinear):
                m.bias.data.zero_()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def make_sigma_parameters(self):
        sigma_parameters = []
        for k, v in self.named_parameters():
            if 'sigma' in k:
                sigma_parameters.append(v)
        return sigma_parameters

    def make_phi_parameters(self):
        phi_parameters = []
        for k, v in self.named_parameters():
            if 'phi' in k:
                phi_parameters.append(v)
        return phi_parameters

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

    def forward(self, input):
        return self.f(input)

def resnet9_dec(data_shape=[3,32,32], target_size=10, ):
    model = ResNet(data_shape, [64, 128, 256, 512], Block, [1, 1, 1, 1], target_size)
    return model


def resnet18_dec(data_shape=[3,32,32], target_size=10, ):
    model = ResNet(data_shape, [64, 128, 256, 512], Block, [2, 2, 2, 2], target_size)
    return model



if __name__ == '__main__':
    
    model = resnet18_dec()
    b = copy.deepcopy(model)
    for name, parameter in b.named_parameters():
        print(name)