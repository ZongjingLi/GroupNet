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
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # [Perception Model]
        self.resolution = config.resolution
        self.W, self.H = self.resolution
        self.grouper = model_dict[config.perception_model_name](resolution = config.resolution, channel_dim = config.channel_dim)

        # [Central Knowledge Executor]
        self.central_executor = CentralExecutor(domain, config.concept_type, config.concept_dim)
        self.relation_encoder = None

        self.affinities = nn.ModuleDict()
        self.affinity_indices = {}


        # [Component Affinity Adapter]
        max_component_num = 32
        component_key_dim = 64 # the key-query dim of the component affinities, combine using the dot product
        feature_map_dim = 128 # the size of F_ij a.k.a each local feature dim, use this as the condition for the attention
        self.feature_map_dim = feature_map_dim
        """MLP as the encoder to encode the edge conditions"""
        self.mlp_encoder = FCBlock(128, 2 ,feature_map_dim * 2, component_key_dim)

        """use embeddings to define the weights to calculate"""
        self.embeddings = nn.Embedding(max_component_num, component_key_dim)
        self.bias_predicter = nn.ModuleDict()
       
        """add some predefined affinities like Spatial Proximity and Spelke Affinity"""

        self.gamma = 0.0
        self.tau = 0.2

    def clear_components(self):
        self.bias_predicter = nn.ModuleDict()
        self.affinities = nn.ModuleDict()
        self.affinity_indices = {}

    def add_spatial_affinity(self):
        self.affinities["spatial_proximity"] = SpatialProximityAffinityCalculator()
        self.affinity_indices["spatial_proximity"] = 0
        self.bias_predicter["spatial_proximity"] =  FCBlock(128, 2, self.feature_map_dim * 2, 1, activation= "nn.GELU()")

    def add_spelke_affinity(self):
        self.affinities["spelke"] = SpelkeAffinityCalculator()
        self.affinity_indices["spelke"] = 1
        self.bias_predicter["spelke"] =  FCBlock(128, 2, self.feature_map_dim * 2, 1, activation= "nn.GELU()")

    def toggle_component(self, name, freeze = True):
        self.affinities[name].requires_grad_(not freeze)
    
    def toggle_component_except(self, name, freeze = True):
        for key in self.affinities:
            if name != key: self.affinities[key].requires_grad_(not freeze)
            else: self.affinities[key].requires_grad_(freeze)

    def toggle_motion(self, freeze = True):
        assert "spelke" in self.affinities, "spelke affinity is not contained in the model"

    def freeze_components(self, freeze = True):
        for key in self.affinities: self.affinities[key].requires_grad_(not freeze)
    
    def add_affinities(self, affinity_names: List[str]):
        """set up general affinity calculators for each name, custom version not included"""
        for i,name in enumerate(affinity_names):
            self.affinities[name] = GeneralAffinityCalculator(name)
            self.affinity_indices[name] = i + len(self.affinity_indices)
            self.bias_predicter[name] = FCBlock(128, 2, self.feature_map_dim * 2, 1, activation= "nn.GELU()")
        
    def visual_query_grounding(self, ims, programs, answers = None, target = None, save_path = None, save_idx = None):
        """predicte the object affnity using the logits"""
        percept_outputs = self.calculate_object_affinity(ims, target = target, working_resolution = (self.W, self.H), verbose = False)
        obj_affinity = percept_outputs["affinity"]
        indices = percept_outputs["indices"]
        
        """extract masks from the predicted object affinity logits"""
        predict_masks, _ ,alive, prop_maps = self.extract_segments(obj_affinity, indices)
        predict_masks = torch.einsum("bwhn,bnd->bwhn", predict_masks, alive)

        """generate the query featueres and set information to ground by language"""
        B, W, H, M = predict_masks.shape
        alive = torch.ones_like(alive)
        rand_features = torch.randn([B, M, 100])

        query_outputs = self.language_grounding(
        percept_outputs, programs, answers ,masks = predict_masks, alive = alive, features = rand_features)
        return query_outputs

 
    def calculate_object_affinity(self, 
                                  img, 
                                  targets: torch.Tensor = None,
                                  working_resolution = (128,128),
                                  keys : List[str] = None,
                                  augument: dict = {},
                                  verbose = False):
        outputs = {}
        device = self.device
        B, C, W_, H_ = img.shape
        W, H = working_resolution
        img = img.to(device)
        if targets is not None: targets = targets.to(device)
        """step 1: calculate the attention based on the component affinity key"""

        if keys is None: keys = [key for key in self.affinities]
        gather_embeds = torch.cat(
            [self.embeddings(torch.tensor(self.affinity_indices[key]).unsqueeze(0).to(device)).unsqueeze(1) for key in keys],
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

        # this is the for the component wise bias
        edge_bias = [self.bias_predicter[key](edge_features).unsqueeze(1).squeeze(-1) for key in self.affinities]
        edge_bias = torch.cat(edge_bias, dim = 1) 

        if verbose:print("gather affinitie shape:",gather_affinities.shape)
        if verbose:print("edge condition, gather embeds shape:",edge_conditions.shape, gather_embeds.shape)
        edge_conditions = F.normalize(edge_conditions, p=2, dim = -1)
        gather_embeds = F.normalize(gather_embeds, p=2, dim = -1)
        attn = torch.einsum("bnkd,bmd->bmnk", edge_conditions, gather_embeds)
        attn = torch.sigmoid((attn - self.gamma)/self.tau)
        attn = torch.ones_like(attn)
        #attn = torch.softmax(attn * 5, dim = 1)
        
        if verbose:print("attn", list(attn.shape), float(attn.max()), float(attn.min()))

        obj_affinity = torch.einsum("bmnk,bmnk->bnk", attn, gather_affinities - edge_bias * 0)
        
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
        outputs["component_affinity"] = {}
        for i, key in enumerate(self.affinities):
            outputs["component_affinity"][key] = gather_affinities[:,i,:,:]
        outputs["indices"] = indices
        return outputs
    
    def language_grounding(self,percept_outputs,  programs, answers = None, masks = None, alive = None, features = None):
        #TODO: mask feature extractor is not implemented yet
        obj_affinity = percept_outputs["affinity"]
        indices = percept_outputs["indices"]
        component_affinity = percept_outputs["component_affinity"]
        component_affinity["object"] = obj_affinity

        if alive is None and masks is None and features is None:
            predict_masks, _ ,alive, prop_maps = self.extract_segments(obj_affinity, indices)
            predict_masks = torch.einsum("bwhn,bnd->bwhn", predict_masks, alive)
            #features = torch.randn([predict_masks.shape[0], predict_masks.shape[-1], 100])
        else: predict_masks = masks
        M = predict_masks.shape[-1]

        M = predict_masks.shape[-1]
        B, N, K = obj_affinity.shape
        """iterate over batched programs and answer pairs"""
        query_loss = 0.0
        predict_answers = []
        numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
        for batch_idx in range(len(programs[0])):
            context = {
                0:{
                    "end": logit(alive.reshape([B,M])[batch_idx,:]),
                    "features": features[batch_idx,:,:],
                    "masks": predict_masks.reshape([B, N, M])[batch_idx, :, :],
                    "affinity": component_affinity,
                    "executor": self.central_executor}
            }

            query_loss = 0.0
            batch_answers = []
            for query_idx in range(len(programs)):
                program = programs[query_idx][batch_idx]
                
                results = self.central_executor.evaluate(program, context)

                gt_answer = answers[query_idx][batch_idx]

                if gt_answer in ["yes", "no"]:
                    #print(program, results["end"])
                    if gt_answer == "yes":
                        query_loss +=  -1. * torch.log(results["end"].sigmoid())
                    else:
                         query_loss +=  -1. * torch.log( 1 - results["end"].sigmoid())
                    batch_answers.append("yes" if results["end"] > 0. else "no")
                elif gt_answer in numbers:
                    batch_answers.append(int(results["end"] + 0.5))
                else:
                    #assert gt_answer in results["end"], f"unknown grounding gt answer: {gt_answer}"
                    #query_loss += -1. * torch.log(results["end"][gt_answer].sigmoid())
                    batch_answers.append(results["end"])
            predict_answers.append(batch_answers)
        query_loss /= (len(programs) * len(programs[0]))
                
        return {"loss": query_loss, "answers":predict_answers}

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
    
    def predict_masks(self, img, resolution = None, augument = {}):
        if resolution is None: resolution = self.resolution
        outputs = self.calculate_object_affinity(img, working_resolution = resolution, augument = augument)
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