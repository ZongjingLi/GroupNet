'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-06-05 18:17:54
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-06-05 18:17:55
 # @ Description: This file is distributed under the MIT license.
'''
import torch
import torch.nn as nn

import taichi as ti
import taichi.math as tm
ti.init(arch = ti.gpu)

ti_img = ti.types.matrix(128, 128, int)
mat4x3i = ti.types.matrix(4, 3, int)

@ti.kernel
def linear_dht_transform(params : int):
    return 0

@ti.data_oriented
class LinearDHT(nn.Module):
    def __init__(self, num_r, num_t, resolution = (128,128)):
        super().__init__()
        self.W, self.H = resolution
        self.num_r, self.num_t = num_r, num_t

        self.params = ti.field(dtype = ti.f32, shape = (num_r, num_t))
        self.img =  ti.field(dtype = ti.f32, shape = (self.W, self.H))
    
    def set_img(self, img):
        self.img.from_torch(img)
 
    @ti.kernel
    def localize_parameters(self):
        for r, t in self.params:
            self.params[r, t] += self.img[x,y]

if __name__ == "__main__":
    import cv2 as cv
    import matplotlib.pyplot as plt
    img = cv.imread('/Users/melkor/Documents/datasets/Plagueworks/train/img/img_54_1_a.png',0)
    img = cv.resize(img, (128,128))

    edges = cv.Canny(img,100,200)
    boundary = torch.tensor(edges).unsqueeze(0) /255.
    plt.subplot(121)
    plt.imshow(img / 255.)
    plt.subplot(122)
    plt.imshow(boundary[0])

    img = torch.tensor(img) / 255.
    print(img.shape)

    line_dht = LinearDHT(100, 100)

    line_dht.set_img(boundary[0])
    line_dht.localize_parameters()

    plt.figure("parameter space")
    plt.imshow(line_dht.params.to_numpy())
    plt.show()