'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-04-23 07:52:32
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-04-23 07:52:34
 # @ Description: This file is distributed under the MIT license.
'''

from mvcl.custom import SpatialProximityAffinityCalculator,\
    SpelkeAffinityCalculator, GeneralAffinityCalculator

from datasets.sprites_base_dataset import SpritesBaseDataset

def group_affinity(self, img, affinity):
    pass

if __name__ == "__main__":
    albedo_affinity = GeneralAffinityCalculator("albedo")