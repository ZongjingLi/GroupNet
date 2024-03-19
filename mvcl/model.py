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
        self.perception = model_dict[config.perception_model_name](config)

        # [Central Knowledge Executor]
        self.central_executor = CentralExecutor(domain, config)

        # [Neuro Implementations]
        self.implementations = nn.ModuleDict()
    
    def get_mapper(self,name):
        for map_name in self.implementations:
            if map_name == name: return self.implementations[map_name]
        assert False, f"there is no such mapper {name}"
    
    def precompute_concept_features(self, indices):
        """calculate the concept affinities A^c_{i,j}
        Inputs:
            indices:
        Returns:
            a diction that corresponds different concept affinities.
        """
        return 
    
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
            #return logit( torch.softmax(masks, dim = -1)[:,:,values.index(c2)] )
            return logit(masks[:, :, values.index(c2)])
        #return self.central_executor.entailment(c1,c2)
    
    def group_concepts(self, img, concept, key = None, target = None):
        affinity_calculator = self.implementations[concept]

        outputs = self.perception(img, affinity_calculator, key, target_masks = target)

        return outputs

    def segment(self, indices,  affinities):
        masks, agents, alive, propmaps = self.perception.compute_masks(affinities, indices)
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