#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : visual_model.py
# Author : Yiqi Sun
# Email  : rintfd@163.com
# Date   : 6/03/2022
#
# This file is part of MecThuen.
# Distributed under terms of the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.distributions as dists

import torchvision

from collections import namedtuple
import os

from numpy.random import random_integers


class Monet(nn.Module):
    def __init__(self, height, width,channel,base = 64):
        super().__init__()
        self.channel = channel
        self.attention = AttentionNet(3,base)
        self.encoder = EncoderNet(height, width,self.channel + 1,base * 2)
        self.decoder = DecoderNet(height, width,base,self.channel)
        self.beta = 0.5
        self.gamma = 0.25
        self.base = base
        self.num = 4
        

    def forward(self, x):
        scope = torch.ones_like(x[:, 0:1])
        masks = []
        for i in range(7-1):
            mask, scope = self.attention(x, scope)
            masks.append(mask)
        masks.append(scope)
        
        loss = torch.zeros_like(x[:, 0, 0, 0])
        mask_preds = []
        full_reconstruction = torch.zeros_like(x)
        
        p_xs = torch.zeros_like(loss)
        kl_zs = torch.zeros_like(loss)
        for i, mask in enumerate(masks):
            z, kl_z = self.__encoder_step(x, mask)
            sigma = 0.2 if i == 0 else 0.8
            p_x, x_recon, mask_pred = self.__decoder_step(x, z, mask, sigma)
            mask_preds.append(mask_pred)
            loss += -p_x + self.beta * kl_z
            p_xs += -p_x
            kl_zs += kl_z
            full_reconstruction += mask * x_recon

        masks = torch.cat(masks, 1)
        tr_masks = torch.transpose(masks, 1, 3)
        q_masks = dists.Categorical(probs=tr_masks)
        q_masks_recon = dists.Categorical(logits=torch.stack(mask_preds, 3))
        kl_masks = dists.kl_divergence(q_masks, q_masks_recon)
        kl_masks = torch.sum(kl_masks, [1, 2])
        # print('px', p_xs.mean().item(),
        #       'kl_z', kl_zs.mean().item(),
        #       'kl masks', kl_masks.mean().item())
        loss += self.gamma * kl_masks
        return {'loss': loss,
                'masks': masks,
                'reconstructions': full_reconstruction}


    def __encoder_step(self, x, mask):
        encoder_input = torch.cat((x, mask), 1)
        q_params = self.encoder(encoder_input)
        means = torch.sigmoid(q_params[:, :self.base]) * 6 - 3 
        sigmas = torch.sigmoid(q_params[:, self.base:]) * 3
        dist = dists.Normal(means, sigmas)
        dist_0 = dists.Normal(0., sigmas)
        z = means + dist_0.sample()
        q_z = dist.log_prob(z)
        kl_z = dists.kl_divergence(dist, dists.Normal(0., 1.))
        kl_z = torch.sum(kl_z, 1)
        return z, kl_z

    def __decoder_step(self, x, z, mask, sigma):
        decoder_output = self.decoder(z)
        x_recon = torch.sigmoid(decoder_output[:, :self.channel])
        mask_pred = decoder_output[:, self.channel - 1]
        dist = dists.Normal(x_recon, sigma)
        p_x = dist.log_prob(x)
        p_x *= mask
        p_x = torch.sum(p_x, [1, 2, 3])
        return p_x, x_recon, mask_pred
    