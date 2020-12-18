import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from .network_utils import AddCoords

def conv1xNxN(in_planes, out_planes, stride=1, ksize = 3, padding=None):
    """1x3x3 convolution with padding"""
    if padding is None:
        padding = ksize//2
    return nn.Conv3d(in_planes, out_planes, kernel_size=[1, ksize, ksize], stride=[1, stride, stride],
                     padding=[0, padding, padding], bias=False)

def convNx1x1(in_planes, out_planes, stride=1, ksize = 3, padding=None):
    """3x1x1 convolution with padding"""
    if padding is None:
        padding = ksize//2
    return nn.Conv3d(in_planes, out_planes, kernel_size=[ksize, 1, 1], stride=[stride, 1, 1],
                     padding=[padding, 0, 0], bias=False)

def conv1x1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlockST(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                norm_layer=None, activation=None):
        
        super(BasicBlockST, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        if activation is None:
            activation = nn.ReLU(inplace=True)

        self.activation = activation
        self.downsample = downsample
        self.stride = stride

        self.bn1a = norm_layer(inplanes)
        self.conv1a = conv1xNxN(inplanes, planes, stride)
        

        self.bn1b = norm_layer(planes)
        self.conv1b = convNx1x1(planes, planes, stride)
        
        self.bn2a = norm_layer(planes)
        self.conv2a = conv1xNxN(planes, planes)
        
        self.bn2b = norm_layer(planes)
        self.conv2b = convNx1x1(planes, planes)

    def forward(self, x):
        identity = x

        out = self.bn1a(x)
        out = self.activation(out)
        out = self.conv1a(out)
        
        out = self.bn1b(out)
        out = self.activation(out)
        out = self.conv1b(out)
        
        out = self.bn2a(out)
        out = self.activation(out)
        out = self.conv2a(out)
        
        out = self.bn2b(out)
        out = self.activation(out)
        out = self.conv2b(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        return out


class ResNet2(nn.Module):

    def __init__(self, layers, block=None, num_classes=50, width_per_group=64, 
                 norm_layer=None, input_channels=1, do_coordconv = False, 
                 fc_shortcut = False, activation = 'relu', init_width = 64):

        super(ResNet2, self).__init__()
        
        self.do_coordconv = do_coordconv
        self.fc_shortcut = fc_shortcut

        if block is None:
            block = BasicBlockST

        if activation == 'relu':
            activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            activation = nn.ELU(inplace=True)
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d

        self._norm_layer = norm_layer
        self.activation = activation

        self.inplanes = init_width
        self.base_width = width_per_group
        
        extra_layers = 0

        if self.do_coordconv:
            extra_layers = 3
            self.add_coords = AddCoords(3)
        

        self.conv1a = conv1xNxN(input_channels, self.inplanes, ksize=7)
        self.bn1b = norm_layer(self.inplanes)
        self.conv1b = convNx1x1(self.inplanes, self.inplanes-extra_layers, ksize=3)

        if self.fc_shortcut:
            self.avgpool0 = nn.AdaptiveAvgPool3d((2, 2, 2))

        self.layer1 = self._make_layer(block, init_width, layers[0])
        self.layer2 = self._make_layer(block, 2*init_width, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4*init_width, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8*init_width, layers[3], stride=2)

        self.bn_fin = norm_layer(8*init_width)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(8*init_width * block.expansion, num_classes)

        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                # norm_layer(planes * block.expansion),
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
            )

        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer, activation = self.activation))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):            
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer, activation = self.activation))

        return nn.Sequential(*layers)


    def _forward_impl(self, x):
        # See note [TorchScript super()]
        
        x = self.conv1a(x)
        x = self.bn1b(x)
        x = self.activation(x)
        x = self.conv1b(x)

        if self.do_coordconv:
            x = self.add_coords(x)

        if self.fc_shortcut:
            skip = self.avgpool0(x)
            skip = torch.flatten(skip, 1)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.bn_fin(x)
        x = self.activation(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        if self.fc_shortcut:
            x += skip

        

        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)