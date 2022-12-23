import torch
import torch.nn as nn
import math
from typing import Optional
from torch import nn, Tensor

class InstanceL2Norm(nn.Module):
    """Instance L2 normalization.
    """
    def __init__(self, size_average=True, eps=1e-5, scale=1.0):
        super().__init__()
        self.size_average = size_average
        self.eps = eps
        self.scale = scale

    def forward(self, input):
        if self.size_average:
            return input * (self.scale * ((input.shape[1] * input.shape[2] * input.shape[3]) / (
                        torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps)).sqrt())  # view
        else:
            return input * (self.scale / (torch.sum((input * input).reshape(input.shape[0], 1, 1, -1), dim=3, keepdim=True) + self.eps).sqrt())

def _get_clones(module, N):
    # return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
    return nn.ModuleList([module for i in range(N)])

class TransformerEncoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model):
        super().__init__()
        self.self_attn = multihead_attn
        # Implementation of Feedforward model
        self.FFN = FFN
        norm_scale = math.sqrt(1.0 / (d_model * 4 * 4))
        self.norm = InstanceL2Norm(scale=norm_scale)

    def instance_norm(self, src, input_shape):
        num_imgs, batch, dim, h, w = input_shape
        # Normlization
        src = src.reshape(num_imgs, h, w, batch, dim).permute(0, 3, 4, 1, 2)
        src = src.reshape(-1, dim, h, w)
        src = self.norm(src)
        # reshape back
        src = src.reshape(num_imgs, batch, dim, -1).permute(0, 3, 1, 2)
        src = src.reshape(-1, batch, dim)
        return src

    def forward(self, src, input_shape, pos: Optional[Tensor] = None):
        # query = key = value = src
        query = src
        key = src
        value = src

        # self-attention
        src2 = self.self_attn(query=query, key=key, value=value)
        src = src + src2
        src = self.instance_norm(src, input_shape)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model=512, num_encoder_layers=6, activation="relu"):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(multihead_attn, FFN, d_model)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

    def forward(self, src, pos: Optional[Tensor] = None):
        assert src.dim() == 5, 'Expect 5 dimensional inputs'
        src_shape = src.shape
        num_imgs, batch, dim, h, w = src.shape

        src = src.reshape(num_imgs, batch, dim, -1).permute(0, 3, 1, 2)
        src = src.reshape(-1, batch, dim)

        if pos is not None:
            pos = pos.view(num_imgs, batch, 1, -1).permute(0, 3, 1, 2)
            pos = pos.reshape(-1, batch, 1)

        output = src

        for layer in self.layers:
            output = layer(output, input_shape=src_shape, pos=pos)

        # [L,B,D] -> [B,D,L]
        # output_feat = output.reshape(num_imgs, h, w, batch, dim).permute(0, 3, 4, 1, 2)
        # output_feat = output_feat.reshape(-1, dim, h, w)
        return output#, output_feat
