'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-01-10 07:33:47
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-01-10 07:36:01
 # @ Description: This file is distributed under the MIT license.
'''
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
from rinarak.utils.os import load_json, save_json

class SpritesBaseDataset(Dataset):
    def __init__(self, split = "train", data_dir = "/Users/melkor/Documents/datasets"):
        super().__init__()
        self.split = split
        self.data_dir = data_dir
        self.im_path = data_dir + "/sprites_env/base/{}/{}.png"
        self.mask_path = data_dir + "/sprites_env/base/{}/{}.npy"
        self.annotation_path = data_dir + "/sprites_env/base/{}/{}.json"
    
    def __len__(self): return 250

    def __getitem__(self, idx):
        data = {}
        img = torch.tensor(plt.imread(self.im_path.format(self.split,idx)))
        masks = np.load(self.mask_path.format(self.split,idx))
        annotations = load_json(self.annotation_path.format(self.split, idx))
        data["img"] = normal_img(img)
        data["masks"] = masks

        questions = annotations["questions"]
        programs = annotations["programs"]
        answers = annotations["answers"]
        data["programs"] = programs[:-2]
        data["questions"] = questions[:-2]
        data["answers"] = answers[:-2]
        return data

exist_template = f"""
    (exists
        (Pr (color))
    )
"""

compound_exist_forall_template = f"""
    (exists
        (intersect
            (exists
                (Pr (color (expand (scene $0)) ) {{}}) 
            )
            (forall
                (Pr (shape (expand (scene $0)) ) {{}})
            )
        )
    )
"""

def generate_sp_exists(language_annotations, scene_annotation, values):
    program = f"""
        (exists
            (intersect 
                (scene $0)
                (forall
                    (union
                        (Pr ({{}} (expand (scene $0)) ) {{}})
                        (not(expand (scene $0))) 
                    )
                )
            )
        )
        """
    for key in values:
        valid_values = values[key]
        sample_value = np.random.choice(valid_values)
        sample_program = program.format(key, sample_value)
        answer = "yes" if sample_value in scene_annotation[key] else "no"
        language = f"is there any object with {key} of {sample_value}"
        language_annotations["questions"].append(language)
        language_annotations["programs"].append(sample_program)
        language_annotations["answers"].append(answer)

def generate_sp_double_exists(language_annotations, scene_annotation, values):
    program = f"""
        (exists
            (intersect 
                (intersect
                    (forall
                        (union
                            (Pr ({{}} (expand (scene $0)) ) {{}})
                            (not(expand (scene $0))) 
                        )
                    )
                    (forall
                        (union
                            (Pr ({{}} (expand (scene $0)) ) {{}})
                            (not(expand (scene $0))) 
                        )
                    )
                )
                (scene $0)
            )
        )
        """
    for key1 in ["color"]:
        sample_value1 = np.random.choice(values[key1])
        for key2 in ["shape"]:
            sample_value2 = np.random.choice(values[key2])

            sample_program = program.format(key1, sample_value1, key2, sample_value2)
            flag = sample_value1 in scene_annotation["color"] and sample_value2 in scene_annotation["shape"]
            answer = "yes" if flag else "no"
            language = f"is there any object with {key1} of {sample_value1} and with {key2} of {sample_value2}"
            language_annotations["questions"].append(language)
            language_annotations["programs"].append(sample_program)
            language_annotations["answers"].append(answer)

def generate_counts(language_annotations, scene_annotation, values):
    query = "how many objects has {} components."
    program = f"""
        (count  
                (intersect
                (forall
                    (union
                        (Pr ({{}} (expand (scene $0)) ) {{}})
                        (not(expand (scene $0))) 
                    )
                )
                (scene $0)
                )
        )
        """
    for key in values:
        valid_values = values[key]
        sample_value = np.random.choice(valid_values)
        sample_program = program.format(key, sample_value)
        answer = 0
        for value in scene_annotation[key]:
            if value == sample_value: answer += 1
        language = f"how many objects with {key} of {sample_value} are there?"
        language_annotations["questions"].append(language)
        language_annotations["programs"].append(sample_program)
        language_annotations["answers"].append(str(answer))

def generate_sprites(num_scenes = 10, resolution = (64,64), split = "train", data_dir = "/Users/melkor/Documents/datasets"):
    max_num_objs = 4
    resolution = resolution
    im_path = data_dir + "/sprites_env/base/{}/{}.png"
    mask_path = data_dir + "/sprites_env/base/{}/{}"
    values = {  "color": ["red","green","blue","not-any-color"],
                "shape":["square","circle","diamond", "not-any-shape"]
                }
    for scene_id in range(num_scenes):
        scene = {}
        
        num_objs = random.randint(1,max_num_objs) # include the interval ends
        width, height = resolution
        canvas = np.zeros([width,height,3])
        masks = np.zeros([width, height])
        scene_annotation = {"color":[],"shape":[]}

        # background information
        #scene_annotation["color"].append("not-any-color")
        #scene_annotation["shape"].append("not-any-shape")
        for idx in range(num_objs):

            # choose the size of the sprite
            pos_x = random.randint(0, width - 12)
            pos_y = random.randint(0, height - 12)
            scale = random.randint(12, min(14, height-pos_y, width-pos_x))

            # choose shape of the sprite
            shape = random.choice(["square","diamond","circle"])
            
            # choose the color of the spirte
            color = random.randint(0,2)
            scene_annotation["color"].append(values["color"][color])
            
            # render the sprite on the canvas and mask
            if shape == "circle":  # draw circle
                scene_annotation["shape"].append("circle")
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
                scene_annotation["shape"].append("square")
                noise = np.random.uniform(0.0,0.2)
                canvas[pos_x:pos_x + scale, pos_y:pos_y + scale, color] = 1.0 - noise
                masks[pos_x:pos_x + scale, pos_y:pos_y + scale] = 1 + idx
            else:  # draw square turned by 45 degrees
                scene_annotation["shape"].append("diamond")
                center_x = pos_x + scale // 2.0
                center_y = pos_y + scale // 2.0
                for x in range(height):
                    for y in range(width):
                        noise = np.random.uniform(0.0,0.2)
                        if abs(x - center_x) + abs(y - center_y) < (scale // 2.0):
                            canvas[x][y][color] = 1.0 - noise
                            masks[x][y] = idx + 1
        plt.imsave(im_path.format(split,scene_id),canvas)
        np.save(mask_path.format(split,scene_id),masks)
        # generate language groundings for the sprites dataset.
        language_annotations = {}
        language_annotations["questions"] = []
        language_annotations["answers"] = []
        language_annotations["programs"] = []
        # Type I: existential quantification quries.
        for i in range(12):
            generate_sp_exists(language_annotations, scene_annotation, values);
        # Type II: compound extistenial quantification.
        generate_sp_double_exists(language_annotations, scene_annotation, values)
        # Type III: counting based quries.
        for i in range(6):
            generate_counts(language_annotations, scene_annotation, values)
        save_json(language_annotations,data_dir + "/sprites_env/base/{}/{}.json".format(split,scene_id))
    return 



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--command",        default = "generate")

    args = parser.parse_args()

    if args.command == "generate":
        generate_sprites(1200, split = "train")
