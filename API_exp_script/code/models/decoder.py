"""
Decoder for few shot segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Refine(nn.Module):
    def __init__(self, inplanes, planes, scale_factor=2):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3,3), padding=(1,1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)
        self.scale_factor = scale_factor

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[-2:], mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, indim, mdim, out_dim, dp_r=0.5):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(indim, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(int(indim/4), mdim)  # 1/8 -> 1/4
        self.RF2 = Refine(int(indim/8), mdim)  # 1/4 -> 1
        self.pred2 = nn.Conv2d(mdim, out_dim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.dropout=nn.Dropout(p=dp_r)

    def forward(self, x, prototypes, skip_inputs, index): #
        m4 = self.dropout(self.ResMM(self.convFM(torch.cat((x, prototypes), 1))))
        m3 = self.dropout(self.RF3(skip_inputs[0][:,index], m4))  # out: 1/8, 256
        m2 = self.RF2(skip_inputs[1][:,index], m3)  # out: 1/4, 256

        p = self.pred2(F.relu(m2))

        # p = F.interpolate(p2, scale_factor=4, mode='bilinear', align_corners=False)
        return p  # , p2, p3, p4
