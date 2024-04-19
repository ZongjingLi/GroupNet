'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-01-24 17:50:01
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-01-24 18:10:38
 # @ Description: This file is distributed under the MIT license.
 '''

import torch
import torch.nn as nn
import torch.nn.functional as F
from rinarak.dklearn.nn.mlp import FCBlock

from typing import NamedTuple, List
from .config import *
from .custom import GeneralAffinityCalculator, SpatialProximityAffinityCalculator, SpelkeAffinityCalculator
from .percept.metanet import weighted_softmax

model_dict = {
    "MetaNet": MetaNet,
    "PropNet": None,
}

class MetaVisualLearner(nn.Module):
    def __init__(self, domain, config):
        super().__init__()
        self.config = config
        self.domain = domain

        # [Perception Model]
        self.resolution = config.resolution
        self.W, self.H = self.resolution
        self.grouper = model_dict[config.perception_model_name](resolution = config.resolution, channel_dim = config.channel_dim)

        # [Central Knowledge Executor]
        self.central_executor = None#CentralExecutor(domain, config)
        self.affinities = nn.ModuleDict()
        self.affinity_indices = {}


        # [Component Affinity Adapter]
        max_component_num = 999
        component_key_dim = 64 # the key-query dim of the component affinities, combine using the dot product
        feature_map_dim = 128 # the size of F_ij a.k.a each local feature dim, use this as the condition for the attention
        self.embeddings = nn.Embedding(max_component_num, component_key_dim)
        self.mlp_encoder = FCBlock(128,3,feature_map_dim * 2, component_key_dim)
        self.bias_predicter: nn.Module = FCBlock(128, 3, feature_map_dim * 2, 1, activation= "nn.GELU()")

        """add some predefined affinities like Spatial Proximity and Spelke Affinity"""
        self.affinities["spelke"] = SpelkeAffinityCalculator()
        self.affinity_indices["spelke"] = 0

        self.affinities["spatial_proximity"] = SpatialProximityAffinityCalculator()
        self.affinity_indices["spatial_proximity"] = 1
        



    def freeze_components(self, freeze = True):
        for key in self.affinities: self.affinities[key].requires_grad_(not freeze)
    
    def add_affinities(self, affinity_names: List[str]):
        """set up general affinity calculators for each name, custom version not included"""
        for i,name in enumerate(affinity_names):
            self.affinities[name] = GeneralAffinityCalculator(name)
            self.affinity_indices[name] = i + len(self.affinity_indices)
        
 
    def calculate_object_affinity(self, 
                                  img, 
                                  targets: torch.Tensor = None,
                                  working_resolution = (128,128),
                                  keys : List[str] = None,
                                  augument: dict = {},
                                  verbose = False):
        outputs = {}
        B, C, W_, H_ = img.shape
        W, H = working_resolution
        """step 1: calculate the attention based on the component affinity key"""

        if keys is None: keys = [key for key in self.affinities]
        gather_embeds = torch.cat(
            [self.embeddings(torch.tensor(self.affinity_indices[key]).unsqueeze(0)).unsqueeze(1) for key in keys],
            dim = 1)

        # BxNxD: gather embeddings vectors for each of affinity
        indices = self.grouper.get_indices([W,H], B, 1) #[B, 2, WH, K]
        B, _, N, K = indices.shape

        # calculate the augument features fro the backbone and condition for attention
        backbone_features = self.grouper.calculate_feature_map(img).permute(0,2,3,1) # [B, W, H, D]
        B, W, H, D = backbone_features.shape
        augument["features"] = backbone_features
        

        gather_affinities = torch.cat([
            self.affinities[key].calculate_affinity_logits(indices, img, augument).unsqueeze(1) for key in keys
            ], dim = 1)
    
        """step 2: calculate and combine the object affinities using precomputed"""
        backbone_features = backbone_features.reshape([B,N,D])

        x_indices = indices[[0,1],...][-1].reshape([B,N*K]).unsqueeze(-1).repeat(1,1,D)
        y_indices = indices[[0,2],...][-1].reshape([B,N*K]).unsqueeze(-1).repeat(1,1,D)

        x_features = torch.gather(backbone_features, dim = 1, index = x_indices).reshape([B, N, K, D])
        y_features = torch.gather(backbone_features, dim = 1, index = y_indices).reshape([B, N, K, D])
        
        edge_features = torch.cat([x_features, y_features], dim = -1)
        edge_conditions = self.mlp_encoder(edge_features)

        edge_bias = self.bias_predicter(edge_features).squeeze(-1) 
   
        

        if verbose:print("gather affinitie shape:",gather_affinities.shape)
        if verbose:print("edge condition, gather embeds shape:",edge_conditions.shape, gather_embeds.shape)
        edge_conditions = F.normalize(edge_conditions, p=2, dim = -1)
        gather_embeds = F.normalize(gather_embeds, p=2, dim = -1)
        attn = torch.einsum("bnkd,bmd->bmnk", edge_conditions, gather_embeds)
        attn = torch.sigmoid(attn)
        #attn = torch.ones_like(attn)
        #attn = torch.softmax(attn * 5, dim = 1)
        
        if verbose:print("attn", list(attn.shape), float(attn.max()), float(attn.min()))

        obj_affinity = torch.einsum("bmnk,bmnk->bnk", attn, gather_affinities - edge_bias) 
        if verbose:print("object affinity:",list(obj_affinity.shape), float(obj_affinity.max()), float(obj_affinity.min()))

        """step 3: (optional) calculate the loss if we have the ground truth object segments"""
        loss = {}
        if targets is not None: loss["adapter_loss"] = self.calculate_object_adapter_loss(obj_affinity, indices, targets)

        """log all the outputs and return the value diction"""
        outputs["loss"] = loss
        outputs["attn"] = attn
        outputs["affinity"] = obj_affinity
        outputs["indices"] = indices
        return outputs

    def calculate_object_adapter_loss(self, logits, sample_inds, target_masks, size = None):
        if len(target_masks.shape) == 3: target_masks = target_masks.unsqueeze(1)
        if size is None: size = [self.W, self.H]
        B, N, K = logits.shape

        segment_targets = F.interpolate(target_masks.float(), size, mode='nearest')

        segment_targets = segment_targets.reshape([B,N]).unsqueeze(-1).long().repeat(1,1,K)
        if sample_inds is not None:
            samples = torch.gather(segment_targets,1, sample_inds[2,...]).squeeze(-1)
        else:
            samples = segment_targets.permute(0, 2, 1)

        targets = segment_targets == samples
        null_mask = (segment_targets == 0) # & (samples == 0)  only mask the rows
        mask = 1 - null_mask.float()

        # [compute log softmax on the logits] (F.kl_div requires log prob for pred)
        y_pred = weighted_softmax(logits, mask)
        y_pred = torch.log(y_pred.clamp(min=1e-8))  # log softmax
        # [compute the target probabilities] (F.kl_div requires prob for target)
        y_true = targets / (torch.sum(targets, -1, keepdim=True) + 1e-9)
        
        # [compute kl divergence]
        kl_div = F.kl_div(y_pred, y_true, reduction='none') * mask
        kl_div = kl_div.sum(-1)

        # [average kl divergence aross rows with non-empty positive / negative labels]
        agg_mask = (mask.sum(-1) > 0).float()
        loss = kl_div.sum() / (agg_mask.sum() + 1e-9)

        return loss#, y_pred
    
    def predict_masks(self, img, resolution = None):
        if resolution is None: resolution = self.resolution
        outputs = self.calculate_object_affinity(img, working_resolution = resolution)
        obj_affinity = outputs["affinity"]
        indices = outputs["indices"]
        masks, agents, alive, prop_maps = self.extract_segments(obj_affinity, indices)
        return {"masks":torch.einsum("bwhn,bnd->bwhn", masks, alive)}

    def extract_segments(self, logits, indices):
        return self.grouper.compute_masks(logits, indices)
    
    def get_mapper(self,name : str):
        for map_name in self.affinities:
            if map_name == name: return self.affinities[map_name]
        assert False, f"there is no such mapper {name}"
    
    def get_concept_embedding(self, name):return self.central_executor.get_concept_embedding(name)
    
    def entailment(self,c1, c2): 
        """return the entailment probability of c1->c2.
        """
        if isinstance(c2, str):
            parent_type = self.central_executor.get_type(c2)
            values = self.central_executor.type_constraints[parent_type]
            masks = []
            for comp in values:
                masks.append(self.central_executor.entailment(c1,self.get_concept_embedding(comp)).unsqueeze(-1))
            masks = torch.cat(masks, dim = -1)
            masks = F.normalize(masks.sigmoid(), dim = -1)
            return logit(masks[:, :, values.index(c2)])
        #return self.central_executor.entailment(c1,c2)
    
    def group_concepts(self, img, concept, key = None, target = None):
        affinity_calculator = self.affinities[concept]
        outputs = self.grouper(img, affinity_calculator, key, target_masks = target)
        return outputs

    def segment(self, indices,  affinities):
        masks, agents, alive, propmaps = self.grouper.compute_masks(affinities, indices)
        return alive, masks

    def print_summary(self):
        summary_string = f"""
[Perception Model]
perception:{self.config.perception_model_name}  ;; the name of the perception model
resolution:{self.resolution}  ;; working resolution of the object centric perception

[Central Knowlege Base]
concept_type: {self.config.concept_type}    ;; the type of the concept structure
concept_dim: {self.config.concept_dim}      ;; the dim of the concept embedding
object_dim: {self.config.object_dim}        ;; the dim of the object space embedding
"""
        print(summary_string)
        if self.domain is not None:self.domain.print_summary()

def iterative_object_expansion(self, target, affinity):
    """
    input a target mask of part of the image and a set of object affinity, the model iteratively
    add more complex
    """
    return