"""
Encoder for few shot segmentation (VGG16)
"""

import torch
import torch.nn as nn
from .multihead_attention import *
from .transformer import TransformerEncoder

class TfmNet(nn.Module):
    def __init__(self, mdim=512):
        super().__init__()
        self.mdim = mdim
        multihead_attn = MultiheadAttention(feature_dim=1024, n_head=1, key_feature_dim=1024)
        self.tansformer = TransformerEncoder(multihead_attn=multihead_attn, FFN=None, d_model=512,
                           num_encoder_layers=1)
        self.infer = nn.Sequential(
            nn.Linear(mdim, mdim),
            nn.Tanh(),
            nn.Linear(mdim, mdim * 2))

    def forward(self, src):
        atted_src = self.tansformer(src)
        proto = torch.mean(atted_src, dim=0)
        output = self.infer(proto)
        return output[:, :self.mdim], output[:, self.mdim:]


class AmortizedNet(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """
    def __init__(self, mdim=512, deep=False):
        super().__init__()
        self.mdim = mdim

        if deep:
            self.infer = nn.Sequential(
                nn.Linear(mdim, mdim),
                nn.Tanh(),# nn.ELU(),
                nn.Linear(mdim, mdim),
                nn.Tanh(),#nn.ELU(),
                nn.Linear(mdim, mdim * 2))
        else:
            self.infer = nn.Sequential(
                nn.Linear(mdim, mdim),
                nn.ELU(),
                nn.Linear(mdim, mdim * 2))

    def forward(self, x):
        out = self.infer(x)
        return out[0,:self.mdim].unsqueeze(dim=0), out[0,self.mdim:].unsqueeze(dim=0)


def sample(mu, log_std, num):
    return [mu + torch.exp(log_std) * torch.randn(list(mu.size())).cuda() for i in range(num)]

def KL_divergence(mu1, log_std1, mu2, log_std2):
    return torch.sum(
        2*(log_std2 - log_std1)
        + (torch.pow(torch.exp(log_std1), 4)
        + torch.pow(mu1-mu2, 2))/(2*torch.pow(torch.exp(log_std2), 4))
        - 0.5)



