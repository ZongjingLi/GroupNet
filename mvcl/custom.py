'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-02-02 03:21:43
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-02-02 03:21:49
 # @ Description: This file is distributed under the MIT license.
'''
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from rinarak.dklearn.nn.mlp import FCBlock
from rinarak.dklearn.cv.unet import UNet
from mvcl.percept.backbones import ResidualDenseNetwork
import math
from rinarak.utils.tensor import logit

class AffinityCalculator(nn.Module, ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def calculate_affinity_feature(self, indices, img, augument_feature = None):
        """ take the img as input (optional augument feature) as output the joint feature of the affinities"""
    
    @abstractmethod
    def calculate_entailment_logits(self, logits_features):
        """ take the joint affinity feature as input and output the logits connectivity"""
    

class GeneralAffinityCalculator(AffinityCalculator):
    def __init__(self, name : str):
        super().__init__()
        self.name = name
        latent_dim = 128
        kq_dim = 32
        self.ks_map = nn.Linear(latent_dim, kq_dim)
        self.qs_map = nn.Linear(latent_dim, kq_dim)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def calculate_affinity_feature(self, indices, img, augument_feature = None):
        """ take the img as input (optional augument feature) as output the joint feature of the affinities"""
        return
    
    def calculate_entailment_logits(self, indices, logits_features):
        """ take the joint affinity feature as input and output the logits connectivity"""
    
    def ideal_maps(self, annotated_masks):
        """
        input is:
        an dict annotated masks that contains a mask, is a key does not appear in the annotation, consider it's zero
        """
        if self.name in annotated_masks:return annotated_masks[self.name]
        else: return False

    def calculate_affinity_logits(self, indices, img, augument_features = None):
        _, B, N, K = indices.shape
        device = self.device

        if augument_features is not None:
            if "annotated_masks" in augument_features:
                features = self.ideal_maps(augument_features["annotated_masks"])
                if isinstance(features, bool): # for zero features, just return zero logits
                    print("calculated")
                    return logit(torch.zeros([B,N,K])).to(device)
                #print(f"{B}x{N}xD", features.shape)
                features = features.permute(0,2,3,1)
                flatten_features = features.reshape(B,N,-1).to(device)
                flatten_ks = flatten_features
                flatten_qs = flatten_features
                B, N, D = flatten_features.shape
                #print(flatten_features.shape)
            else:
                features = augument_features["features"].reshape(B,N,-1).to(device)
                flatten_ks = self.ks_map(features)
                flatten_qs = self.qs_map(features)
                B, N, D = flatten_ks.shape

        x_indices = indices[[0,1],...][-1].reshape([B,N*K]).unsqueeze(-1).repeat(1,1,D).to(device)
        y_indices = indices[[0,2],...][-1].reshape([B,N*K]).unsqueeze(-1).repeat(1,1,D).to(device)

        # gather image features and flatten them into 1dim features
        x_features = torch.gather(flatten_ks, dim = 1, index = x_indices).reshape([B, N, K, D])
        y_features = torch.gather(flatten_qs, dim = 1, index = y_indices).reshape([B, N, K, D])

        B, N, K, D = x_features.shape
        
        x_features = x_features.reshape([B, N, K, D])
        y_features = y_features.reshape([B, N, K, D])

        if "annotated_masks" in augument_features:
            #logits = logit(x_features == y_features, eps = 1e-6)
            logits = torch.sum( ((x_features - y_features) ** 2) , dim = -1)** 0.5
            eps = 0.01
            #print(logits.max(), logits.min())
            inverse_div = 1 / ( eps + 7.2 * logits.reshape([B, N, K]) )
            #inverse_div = (logits < 0.1).float()
            logits = logit(inverse_div)
            #print(inverse_div.shape, inverse_div.max(), inverse_div.min())
        else:
            logits = (x_features * y_features).sum(dim = -1) * (D ** -0.5)
        logits = logits.reshape([B, N, K])
        return logits

class SpatialProximityAffinityCalculator(AffinityCalculator):
    def __init__(self):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def calculate_affinity_feature(self, indices, img, augument_feature = None):
        """ take the img as input (optional augument feature) as output the joint feature of the affinities"""
        return
    
    def calculate_entailment_logits(self, indices, logits_features):
        """ take the joint affinity feature as input and output the logits connectivity"""
    

    def calculate_affinity_logits(self, indices, img, augument_features = None):
        _, B, N, K = indices.shape
        device = self.device
        D = 2

        """remember to fix the device arguments here and proceed"""
        xs = torch.linspace(-1, 1, int(math.sqrt(N))).to(device)
        ys = torch.linspace(-1, 1, int(math.sqrt(N))).to(device)
        xs, ys = torch.meshgrid([xs,ys])
        xs = xs[None, ..., None].repeat(B, 1, 1, 1)
        ys = ys[None, ..., None].repeat(B, 1, 1, 1)
        flatten_locs = torch.cat([xs, ys], dim = -1).reshape([B, N, D])


        x_indices = indices[[0,1],...][-1].reshape([B,N*K]).unsqueeze(-1).repeat(1,1,D).to(device)
        y_indices = indices[[0,2],...][-1].reshape([B,N*K]).unsqueeze(-1).repeat(1,1,D).to(device)

        # gather image features and flatten them into 1dim features
        x_loc = torch.gather(flatten_locs, dim = 1, index = x_indices).reshape([B, N, K, D]).to(device)
        y_loc = torch.gather(flatten_locs, dim = 1, index = y_indices).reshape([B, N, K, D]).to(device)

        y_loc = torch.ones_like(x_loc) * 0.0

        B, N, K, D = x_loc.shape
        x_loc = x_loc.reshape([B, N, K, D])
        y_loc = y_loc.reshape([B, N, K, D])
        
        """TODO: this spatial proximity is only calculated over the grid points and not in standar scale!"""
        logits = torch.sum( ((x_loc - y_loc) ** 2) , dim = -1)** 0.5

        eps = 0.1
        inv_diff = 1 / ( eps + 150 * logits.reshape([B, N, K]) )
        logits = logit(inv_diff)
        #logits +=  torch.randn_like(logits) * 2.0
        return logits

class SpelkeAffinityCalculator(GeneralAffinityCalculator):
    def __init__(self):
        super().__init__("spelke")

class AutoEncoderAffinityCalculator(AffinityCalculator):
    def __init__(self):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def calculate_affinity_feature(self, indices, img, augument_feature = None):
        """ take the img as input (optional augument feature) as output the joint feature of the affinities"""

    def calculate_entailment_logits(self, indices, logits_features):
        """ take the joint affinity feature as input and output the logits connectivity"""

    def calculate_affinity_logits(self, indices, img, augument_features = None):
        return 0


class ObjectAffinityFeatures(AffinityCalculator):
    def __init__(self, input_dim,  latent_dim):
        super().__init__()
        self.name : str = "object"
        kq_dim = 132
        #assert input_dim % 2 == 0,"input dim should be divisble by 2 as it is a pair of patches features"
        self.backbone = ResidualDenseNetwork(latent_dim, n_colors = input_dim)
        self.ks_map = nn.Linear(latent_dim, kq_dim)
        self.qs_map = nn.Linear(latent_dim, kq_dim)

    def calculate_affinity_feature(self, indices, img, augument_feature = None):
        _, B, N, K = indices.shape

        conv_features = self.backbone(img)
        conv_features = torch.nn.functional.normalize(conv_features, dim = -1, p = 2)
        conv_features = conv_features.permute(0,2,3,1)
        B, W, H, D = conv_features.shape

        flatten_features = conv_features.reshape([B,W*H,D])
        flatten_features = torch.cat([
            flatten_features, # [BxNxD]
            torch.zeros([B,1,D]), # [Bx1xD]
        ], dim = 1) # to add a pad feature to the edge case
        flatten_ks = self.ks_map(flatten_features)
        flatten_qs = self.qs_map(flatten_features)


        x_indices = indices[[0,1],...][-1].reshape([B,N*K]).unsqueeze(-1).repeat(1,1,D)
        y_indices = indices[[0,2],...][-1].reshape([B,N*K]).unsqueeze(-1).repeat(1,1,D)

        # gather image features and flatten them into 1dim features
        x_features = torch.gather(flatten_ks, dim = 1, index = x_indices)
        y_features = torch.gather(flatten_qs, dim = 1, index = y_indices)
        
        x_features = x_features.reshape([B, N, K, D])
        y_features = y_features.reshape([B, N, K, D])
        #rint(x_features.shape, y_features.shape)
        return torch.cat([x_features, y_features], dim = -1)
    
    def calculate_entailment_logits(self, logits_features, key = None):
        B, N, K, D = logits_features.shape
        assert D % 2 == 0, "not a valid feature dim"
        DC = D // 2
        x_features = logits_features[:,:,:,:DC]
        y_features = logits_features[:,:,:,DC:]
        logits = (x_features * y_features).sum(dim = -1) * (DC ** -0.5)
        logits = logits.reshape([B, N, K])
        return logits

class ColorAffinityFeatures(AffinityCalculator):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        assert input_dim % 2 == 0, "input dim should be divisble by 2 as it is a pair of patches features"
    
class CategoryAffinityFeatures(AffinityCalculator):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        assert input_dim % 2 == 0, "input dim should be divisible by 2 as it is a pair of patchs features"

    
class TextureAffinityFeatures(AffinityCalculator):
    def __init__(self,input_dim, output_dim):
        super().__init__()
        assert input_dim % 2 == 0, "input dim should be divisible by 2 as it is a pair of patchs features"



def build_demo_domain(model):
    from .primitives import Primitive
    model.implementations["universal"] = IdMap()#UnivseralMapper(4,132)
    model.implementations["color"] = ColorMapper(4,model.config.object_dim)
    model.implementations["shape"] = ShapeMapper(4,model.config.concept_dim)
    # [Pre-define some concept mapper]
    color = Primitive.GLOBALS["color"]
    color.value = lambda x: {**x, "features": x["model"].get_mapper("color")(x["features"])}
    shape = Primitive.GLOBALS["shape"]
    shape.value = lambda x: {**x, "features": x["model"].get_mapper("shape")(x["features"])}
    return model

def build_line_demo_domain(model):
    from .primitives import Primitive
    model.implementations["universal"] = nn.Linear(1, 32)
    model.implementations["color"] = nn.Linear(32, 1)
    # [Pre-define some concept mappers]
    color = Primitive.GLOBALS["color"]
    color.value = lambda x: {**x, "features": x["model"].get_mapper("color")(x["features"])}
    return model

def build_meta_domain(model, config):
    model.implementations["object"] = ObjectAffinityFeatures(config.channel_dim, 128)
    return model

def build_custom(model, config, domain_name):
    print(config.resolution)
    #if domain_name == "demo": return build_demo_domain(model)
    #if domain_name == "line_demo": return build_line_demo_domain(model)
    if domain_name == "MetaLearn": return build_meta_domain(model, config)