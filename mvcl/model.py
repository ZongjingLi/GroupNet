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

from .config import *

class SetNet(nn.Module):
    def __init__(self,config):
        super().__init__()
        latent_dim = 128
        self.fc0 = nn.Linear(config.channel_dim, latent_dim)
        self.fc1 = nn.Linear(latent_dim, config.object_dim)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def forward(self, x, end = None):
        x = self.fc0(x)
        x = nn.functional.relu(x)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        end = logit(torch.ones(x.shape[1]).to(self.device))
        return end, x

model_dict = {
    "MetaNet": MetaNet,
    "PropNet": None,
    "SetNet": SetNet
}

class MetaVisualLearner(nn.Module):
    def __init__(self, domain, config):
        super().__init__()
        self.config = config
        self.domain = domain

        # [Perception Model]
        self.resolution = config.resolution
        self.perception = model_dict[config.perception_model_name](config)

        # [Central Knowledge Executor]
        self.central_executor = CentralExecutor(domain, config)

        # [Neuro Implementations]
        self.implementations = nn.ModuleDict()
    
    def get_mapper(self,name):
        for map_name in self.implementations:
            if map_name == name: return self.implementations[map_name]
        assert False, f"there is no such mapper {name}"
    
    def get_concept_embedding(self, name):return self.central_executor.get_concept_embedding(name)
    
    def entailment(self,c1, c2): 
        if isinstance(c2, str):
            parent_type = self.central_executor.get_type(c2)
            values = self.central_executor.type_constraints[parent_type]
            masks = []
            for comp in values:
                masks.append(self.central_executor.entailment(c1,self.get_concept_embedding(comp)).unsqueeze(-1))
            masks = torch.cat(masks, dim = -1)
            masks = F.normalize(masks.sigmoid(), dim = -1)
            #return logit( torch.softmax(masks, dim = -1)[:,:,values.index(c2)] )
            return logit(masks[:, :, values.index(c2)])
        #return self.central_executor.entailment(c1,c2)
    
    def segment(self, boolean_map, base_feature_map):
        scores = 1
        masks = 1
        return scores, masks

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