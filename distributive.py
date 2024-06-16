import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
num_pts = 100
generate_t = torch.linspace(-3, 3, num_pts).unsqueeze(-1)
kset_x = torch.cos(generate_t) * torch.exp(generate_t) * 0.1
kset_y = torch.sin(generate_t) * torch.exp(generate_t) * 0.1
dataset = torch.cat([kset_x, kset_y], dim = -1)


def sq_norm(M, k):
    # M: b x n --(norm)--> b --(repeat)--> b x k
    return (torch.norm(M, dim=1)**2).unsqueeze(1).repeat(1,k)

class IdealDenoiser:
    def __init__(self, dataset):
        self.data = torch.stack(list(dataset))

    def __call__(self, x, sigma):
        x = x.flatten(start_dim=1)
        d = self.data.flatten(start_dim=1)
        xb, db = x.shape[0], d.shape[0]
        sq_diffs = sq_norm(x, db) + sq_norm(d, xb).T - 2 * x @ d.T
        weights = torch.nn.functional.softmax(-sq_diffs/2/sigma**2, dim=1)
        return (x - torch.einsum('ij,j...->i...', weights, self.data))/sigma


eps = IdealDenoiser(dataset)

plt.figure("gradient control", figsize = (6,6))
x = torch.tensor((np.random.random([100,2]) - 0.5) * 7.).float()
for itrs in range(2,100):
    level = 1/(1 + itrs)
    grads = eps(x, 0.1)    
    plt.scatter(kset_x, kset_y)
    plt.scatter(x[:,0], x[:,1], c = "orange")
    plt.quiver(x[:,0], x[:,1], -grads[:,0], -grads[:,1])

    x = x - grads * level
    plt.pause(0.1)
    plt.cla()

plt.show()
