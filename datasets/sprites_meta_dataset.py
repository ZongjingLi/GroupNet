'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-01-10 07:33:47
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-01-10 07:36:01
 # @ Description: This file is distributed under the MIT license.
'''
from curses import meta
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import random
import matplotlib.pyplot as plt
try:
    from rinarak.utils.data import normal_img
except:
    def normal_img(img):
        if len(img.shape) == 4:
            if not img.shape[1] in [1,3,4]: return img.permute(0,3,1,2)
        if len(img.shape) == 3:
            if not img.shape[0] in [1,3,4]: return img.permute(2,0,1)
from karanir.utils.os import load_json

from shapely.geometry import Polygon, Point

class SpritesMetaDataset(Dataset):
    def __init__(self, split = "train", data_dir = "/Users/melkor/Documents/datasets"):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.im_path = data_dir + "/sprites_meta/{}/{}.png"
        self.mask_path = data_dir + "/sprites_meta/{}/{}.npy"
    
    def __len__(self): return 500

    def __getitem__(self, idx):
        data = {}
        img = torch.tensor(plt.imread(self.im_path.format(self.split,idx)))
        masks = np.load(self.mask_path.format(self.split,idx))
        data["img"] = normal_img(img)
        data["masks"] = masks
        return data

def generate_meta_sprites(num_scenes = 10, resolution = (64,64), split = "train", meta_category="triangle" ,data_dir = "/Users/melkor/Documents/datasets"):
    max_num_objs = 3
    resolution = resolution
    im_path = data_dir + "/sprites_env/meta/{}/{}/{}.png"
    mask_path = data_dir + "/sprites_env/meta/{}/{}/{}"
    for scene_id in range(num_scenes):
        scene = {}
        num_objs = random.randint(1,max_num_objs) # include the interval ends
        
        width, height = resolution
        canvas = np.zeros([width,height,3])
        masks = np.zeros([width, height])

        for idx in range(num_objs):
            # choose the size of the sprite
            pos_x = random.randint(0, width - 12)
            pos_y = random.randint(0, height - 12)
            scale = random.randint(12, min(14, height-pos_y, width-pos_x))

            # choose shape of the sprite
            shape = random.choice(["square","diamond","circle",meta_category])
            
            # choose the color of the spirte
            color = random.randint(0,2)
            
            # render the sprite on the canvas and mask
            if shape == "circle":  # draw circle

                center_x = pos_x + scale // 2.0
                center_y = pos_y + scale // 2.0
                for x in range(height):
                    for y in range(width):
                        dist_center_sq = (x - center_x)**2 + (y - center_y)**2
                        if  dist_center_sq < (scale // 2.0)**2:
                            noise = np.random.uniform(0.0,0.2)
                            canvas[x][y][color] = 1.0 - noise
                            masks[x,y] = 1 + idx
            elif shape == "square":  # draw square
                noise = np.random.uniform(0.0,0.2)
                canvas[pos_x:pos_x + scale, pos_y:pos_y + scale, color] = 1.0 - noise
                masks[pos_x:pos_x + scale, pos_y:pos_y + scale] = 1 + idx
            elif shape == "triangle":
                p1 = (pos_x - scale//2, pos_y)
                p2 = (pos_x + scale//2, pos_y)
                p3 = (pos_x, pos_y + scale)
                point_list = [p1,p2,p3,p1]
                polygon = Polygon([point[0],point[1]] for point in point_list)
                for x in range(height):
                    for y in range(width):
                        if Point(x,y).within(polygon):
                            canvas[-y][x][color] = 1.0
                            masks[-y,x] = 1 + idx
            else:  # draw square turned by 45 degrees
                center_x = pos_x + scale // 2.0
                center_y = pos_y + scale // 2.0
                for x in range(height):
                    for y in range(width):
                        noise = np.random.uniform(0.0,0.2)
                        if abs(x - center_x) + abs(y - center_y) < (scale // 2.0):
                            canvas[x][y][color] = 1.0 - noise
                            masks[x][y] = idx + 1
            plt.imsave(im_path.format(meta_category,split,scene_id),canvas)
            np.save(mask_path.format(meta_category,split,scene_id),masks)
    return 
    
if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--command",        default = "generate")
    parser.add_argument("--num_scenes",     default = 100)
    parser.add_argument("--category",       default = "triangle")

    args = parser.parse_args()
    category = args.category
    num_scenes = args.num_scenes

    if args.command == "generate":
        sys.stdout.write(f"generate meta-sprites-{category} dataset with {num_scenes} scenes.")
        generate_meta_sprites(num_scenes, meta_category = category)