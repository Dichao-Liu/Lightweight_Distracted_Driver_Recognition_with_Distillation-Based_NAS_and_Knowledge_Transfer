import torch
import torch.nn as nn


def conv(in_planes, out_planes, kernel_size=[3,3], stride=1, padding=[1,1], dilation=1, groups=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(kernel_size[0],kernel_size[1],kernel_size[1]),
                     stride=stride, padding=(padding[0],padding[1],padding[1]), dilation=dilation, groups=groups, bias=False)

class PyConv3D(nn.Module):

    def __init__(self, inplans, planes,pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv3D, self).__init__()
        self.conv2_1 = conv(inplans, planes // 2, kernel_size=[pyconv_kernels[0],pyconv_kernels[0]],
                            padding=[pyconv_kernels[0] // 2,pyconv_kernels[0] // 2],
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 2, kernel_size=[pyconv_kernels[1],pyconv_kernels[1]],
                            padding=[pyconv_kernels[1] // 2,pyconv_kernels[1] // 2],
                            stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


class BasicLayer(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, kernel_size_, group=1, group_=4):
        super(BasicLayer, self).__init__()
        self.convs = nn.Sequential(
            PyConv3D(in_planes, out_planes, pyconv_kernels=[kernel_size, kernel_size_], stride=1, pyconv_groups=[group, group_]),
            nn.BatchNorm3d(out_planes),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1,1,1))

    def forward(self, x):
        x = self.convs(x)
        x = self.pool(x)

        return x


class Searched_Net_3D(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.convs = nn.Sequential(
            BasicLayer(3, 32, kernel_size=11, kernel_size_=3, group=1, group_=1),
            BasicLayer(32, 64, kernel_size=9, kernel_size_=5, group=1, group_=1),
            BasicLayer(64, 128, kernel_size=5,  kernel_size_=3, group=1, group_=1),
            BasicLayer(128, 256, kernel_size=3,  kernel_size_=1, group=1, group_=1)
        )
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, n_class)
        )

    def forward(self, x):
        x = self.convs(x)
        x = self.pool(x).view(x.shape[0], -1)
        x = self.classifier(x)
        return x