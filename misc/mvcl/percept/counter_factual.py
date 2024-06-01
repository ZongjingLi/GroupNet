'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-02-26 13:41:17
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-02-26 13:41:19
 # @ Description: This file is distributed under the MIT license.
'''

import torch
import torch.nn as nn

from torch.autograd import Variable

from .patchify import LinearPatchEmbedding
import torch
import torch.nn as nn

from torch.autograd import Variable
from mvcl.percept.patchify import LinearPatchEmbedding

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Linear(input_size, d_model)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        return output

class TransformerDecoder(nn.Module):
    def __init__(self, output_size, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.pos_decoder = PositionalEncoding(d_model, dropout)
        decoder_layers = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers)
        self.decoder = nn.Linear(d_model, output_size)

    def forward(self, tgt, memory):
        tgt = self.pos_decoder(tgt)
        output = self.transformer_decoder(tgt, memory)
        output = self.decoder(output)
        return output

class DualMaskGenerator:
    def __init__(self, p = 0.0, q = 0.9):
        self.p = p
        self.q = q
    def generate(self, size = (1,16,16)):
        mask1 = torch.randint(0, 100, size) >= (self.p * 100)
        mask2 = torch.randint(0, 100, size) >= (self.q * 100)
        return torch.cat([mask1.float()[...,None], mask2.float()[...,None]], dim = -1)

class CounterFactualWorldModel(nn.Module):
    def __init__(self, resolution = (128,128)):
        """
        a masked autoencoder as the counter factual world model, forced to capture short term actions.
        this is essentially a vmae that use xt and masked xt+d to predict the ground truth xt+d input, then
        several interfaces are provided to read specific visual features from the encoder
        """
        super().__init__()
        self.W, self.H = resolution
        patch_dim = 128
        self.patch_size = (1, 8, 8)
        self.resolution = resolution
        d_model = 128
        # a masked autoencoder as the counter factual world model, forced to capture short term actions.
        self.encoder = TransformerEncoder(input_size = patch_dim, d_model = d_model, nhead = 8, num_layers = 6)
        self.decoder = TransformerDecoder(output_size = d_model, d_model = d_model, nhead = 8, num_layers = 6)
        self.patch_linear_decoder = nn.Linear(d_model, self.patch_size[1] * self.patch_size[2] * 3)
        
        self.mask_generator = DualMaskGenerator()
        """a patchify mask generator, mask the 1-p percent of the whole second frame"""
        self.patch_generator = LinearPatchEmbedding(out_dim = patch_dim, patch_size=self.patch_size) # transform an image into a sequence of vectors of size of the patch

    
    def forward(self, x_0, x_t, mask = None):
        B, C, W, H = x_0.shape
        outputs = {}
        """generate a random mask from normal distribution if mask is not provided"""
        pw = self.W // self.patch_size[1]
        ph = self.H // self.patch_size[1]
        mask = mask if mask is not None else self.mask_generator.generate(size = (B, pw, ph)) # BxWxHx2
        outputs["mask"] = mask.clone()

        mask = mask.reshape([B, pw * pw * 2, 1])

        """mask the next frame and make it """
        input_vids = torch.cat([x_0.unsqueeze(1), x_t.unsqueeze(1)], dim = 1)
       
        patches = self.patch_generator(input_vids)
        masked_patches = patches * mask


        encode_features = self.encoder(masked_patches)
        decode_features = self.decoder(patches, encode_features)

        """decode the rgb values of each patch of video features and calculate the loss"""
        patch_values = self.patch_linear_decoder(decode_features).sigmoid()

        decode_vids = self.patch_generator.patches_to_video(patch_values)

        outputs["loss"] = torch.nn.functional.mse_loss(input_vids[:,1,:,:,:], decode_vids[:,1,:,:,:])
        outputs["recons"] = decode_vids
        outputs["mask_patches"] = input_vids
        return outputs

    def sample_mask(self, size = (64,64), p = 0.1):
        mask = torch.randn(size)
        return mask

    def counter_factual_evolution(self):
        return 

    def estimate_optical_flow(self, x_0, x_t, perturb = False):
        """ estimate the optical flow using the infinitismal-perturbation at each point
        Args:
            img1: a batch of images in the first frame Bx3xWxH
            img2: a batch of images in the second frame Bx3xWxH
        Return:
            the estimated optical flow as 2-d vectors with the shape of BxWxHx2
        """
        assert x_0.shape[1] < 4, f"img1 shape as {list(x_0.shape)}"
        B, C, W, H = x_0.shape
        if perturb:
            pw = self.W // self.patch_size[1]
            ph = self.H // self.patch_size[1]
            mask = mask if mask is not None else self.mask_generator.generate(size = (B, pw, ph)) # BxWxHx2
            mask = self.mask_generator.generate()
            """calculate the clean prediction of the second frame"""

            """calculate the perturbed prediction of the second frame"""

            """find the argmax of the perturbation response"""

            """the respondse location and the perturbed location difference is the optical flow"""
        return

    def movable_object_inference(self):
        return 