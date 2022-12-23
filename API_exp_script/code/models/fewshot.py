"""
Fewshot Semantic Segmentation
"""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import Encoder
from .resnet import model_dict
from .decoder import *
from .vi import *
from .multihead_attention import *
from .transformer import TransformerEncoder
import util.lovasz_losses as L



class FewShotSeg(nn.Module):
    """
    Fewshot Segmentation model

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """
    def __init__(self, encoder, out_dim=2, pretrained_path=None, cfg=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}

        print('-- Backbone: ' + encoder)
        # import pdb
        # pdb.set_trace()
        dict_dim = {'ResNet18': 256,
                    'ResNet101': 1024, }
        self.criterion = nn.CrossEntropyLoss(ignore_index=255)
        # Encoder
        self.encoder = model_dict[encoder]
        # Decoder
        self.decoder = Decoder(indim=dict_dim[encoder]*2, mdim=256, out_dim=out_dim)
        # Inference
        self.inference = AmortizedNet(mdim=dict_dim[encoder], deep=True)
        # Inference_ft
        self.inference_ft = AmortizedNet(mdim=dict_dim[encoder], deep=True)
        # Prior
        self.prior = AmortizedNet(mdim=dict_dim[encoder], deep=True)
        # Prior_ft
        self.prior_ft = TfmNet(mdim=dict_dim[encoder])

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs, qry_labels, train=True,n_sample_pro=3, n_sample_mk=4):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0),], dim=0)
        img_fts, skip_fts = self.encoder(imgs_concat)
        fts_size = img_fts.shape[-2:]

        supp_fts = img_fts[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # Wa x Sh x B x C x H' x W'
        qry_fts = img_fts[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)   # N x B x C x H' x W'
        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Wa x Sh x B x H' x W'
        qry_fg_mask = qry_labels.view(
            n_queries, batch_size, *img_size).float()  # N x B x H' x W'

        skip_qry_fts = [i[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *i.shape[-2:])
            for i in skip_fts]

        qry_mus, qry_log_stds = self.prior_ft(qry_fts)
        qry_mus, qry_log_stds = qry_mus.view(n_queries, batch_size, -1), qry_log_stds.view(n_queries, batch_size, -1)

        ###### Compute loss ######
        align_loss = 0
        outputs = []
        kl_losses = 0
        KL_loss_sqs = 0
        KL_loss_qqs = 0
        masks = []
        for epi in range(batch_size):
            ###### Extract prototype ######
            supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            qry_fg_fts = [[self.getFeatures(qry_fts[qry, [epi]],
                                            qry_fg_mask[qry, [epi]])
                           for qry in range(n_queries)]]
            ###### Obtain the prototypes######
            supp_fg_prototypes = self.getPrototype(supp_fg_fts)
            qry_fg_prototypes = self.getPrototype(qry_fg_fts)
            supp_proto_mu, supp_proto_log_std = self.prior(supp_fg_prototypes[0])
            qry_proto_mu, qry_proto_log_std = self.inference((qry_fg_prototypes[0] + supp_fg_prototypes[0])/2.0)
            qry_proto_mu2, qry_proto_log_std2 = self.inference_ft(qry_fg_prototypes[0])
            qry_mu, qry_log_std = qry_mus[0, [epi]], qry_log_stds[0, [epi]]# self.prior_ft(qry_fts[0, [epi]])

            KL_loss_sq = KL_divergence(qry_proto_mu, qry_proto_log_std, supp_proto_mu, supp_proto_log_std)
            KL_loss_qq = KL_divergence(qry_proto_mu2, qry_proto_log_std2, qry_mu, qry_log_std)
            KL_loss_sqs = KL_loss_sqs + KL_loss_sq
            KL_loss_qqs = KL_loss_qqs + KL_loss_qq
            kl_losses = kl_losses + KL_loss_sq + KL_loss_qq

            if train:
                prototypes = sample(qry_proto_mu, qry_proto_log_std, n_sample_pro)
                ft = sample(qry_proto_mu, qry_proto_log_std, n_sample_mk)
                masks_ = self.calMask(qry_fts[:, epi], ft, 1.0)
            else:
                prototypes = sample(supp_proto_mu, supp_proto_log_std, n_sample_pro)
                ft = sample(qry_mu, qry_log_std, n_sample_mk)
                masks_ = self.calMask(qry_fts[:, epi], ft, 1.0)

            prototype_maps = self.tile_prototype(prototypes, fts_size, n_queries)
            masks.append(masks_)

            qry_fts_inputs = torch.cat(
                [torch.cat([qry_fts[:, epi] * m.unsqueeze(dim=1) for m in masks_], 0) for prototype in prototype_maps], 0)
            qry_fts_prototypes =torch.cat(
                [torch.cat([prototype for m in masks_], 0) for prototype in prototype_maps], 0)
            pred = self.decoder(qry_fts_inputs, qry_fts_prototypes, skip_qry_fts, epi)

            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear', align_corners=False))

        # print('sq: ' +  str(KL_loss_sqs/batch_size))
        # print('qq: ' +  str(KL_loss_qqs/batch_size))
        output = torch.stack(outputs, dim=1)  # (N_pro_sample x N_mask_sample x N_query) x B x (1 + Wa) x H x W
        # transpose to B x (N_pro_sample x N_mask_sample x N_query) x (1 + Wa) x H x W
        return output.transpose(0,1), kl_losses/batch_size, masks


    def tile_prototype(self, prototypes, scale, n_q):
        return [i.unsqueeze(2).unsqueeze(3).repeat(n_q, 1, scale[0], scale[1]) for i in prototypes]

    def calMask(self, fts, prototype, scalar=50):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        masks = [torch.sigmoid(F.cosine_similarity(fts, proto[..., None, None], dim=1) * scalar)
                  for proto in prototype]
        # dist = torch.sigmoid(F.cosine_similarity(fts, prototype[..., None, None], dim=1) * scalar)
        return masks


    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling

        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """

        mask = F.interpolate(mask[None, ...], size=fts.shape[-2:], mode='bilinear', align_corners=False)
        masked_fts = torch.sum(fts * mask, dim=(2, 3)) \
                     / (mask.sum(dim=(2, 3)) + 1e-5)  # 1 x C

        # fts = F.interpolate(fts, size=mask.shape[-2:], mode='bilinear')
        # masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
        #     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5) # 1 x C
        return masked_fts


    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        return fg_prototypes

    def train_mode(self):
        self.train()
        for m in self.encoder.modules():  # freeze BN layers
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def loss(self, config, ratio, query_labels, query_pred, kl_loss, n_pro, n_mk):
        query_labels_ = query_labels.unsqueeze(1).unsqueeze(1).repeat(1, n_pro, n_mk, 1, 1) \
            .view([-1] + list(query_labels.shape[-2:]))  # tiled labels for samples
        query_pred_ = query_pred.reshape([-1] + list(query_pred.shape[-3:]))
        query_pred_d1 = query_pred_[:, 1, :, :] - query_pred_[:, 0, :, :]
        query_loss_lovasz = L.lovasz_hinge(query_pred_d1, query_labels, ignore=255)
        loss = self.criterion(query_pred_, query_labels_) * (1.0 - ratio) \
               + query_loss_lovasz * ratio \
               + kl_loss * min(config['kl_loss_scaler'], ratio)

        return loss

