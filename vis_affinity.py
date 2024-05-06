'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-05-04 18:38:21
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-05-04 18:38:27
 # @ Description: This file is distributed under the MIT license.
 '''

import torch
import torch.nn as nn

import tkinter as tk
import os

from PIL import Image

def canvas_click_event(self, event):
    print('Clicked canvas: ', event.x, event.y, event.widget)

width, height = (568,568)

root = tk.Tk()

canvas = tk.Canvas(root, width = width, height = height)
canvas.pack()

img = tk.PhotoImage(file="/Users/melkor/Documents/GitHub/MetaVisualConceptLearner/outputs/props.gif")
canvas.create_image(50, 50, anchor=tk.NW, image = img)

root.mainloop()