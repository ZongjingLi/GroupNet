'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-03-15 12:13:46
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-03-15 12:13:49
 # @ Description: This file is distributed under the MIT license.
'''

import torch
import torch.nn as nn

from rinarak.types import *
from rinarak.program import Program, Primitive
from rinarak.utils.tensor import logit, expat

# [Type Specification] of ObjectSet, Attribute, Boolean and other apsects
Stream = baseType("Stream")
ObjectSet = baseType("ObjectSet")
PrimitiveSet = baseType("PrimitiveSet")
Attribute = baseType("Attribute")
Boolean = baseType("Boolean")
Concept = baseType("Concept")
Integer = baseType("Integer")

# [Return all the objects in the Scene]
operator_scene = Primitive(
    "scene",
    arrow(Stream, ObjectSet),
    lambda x: {**x,"end": x["end"]}
)

# [Existianial quantification, exists, forall]
operator_exists = Primitive(
    "exists",
    arrow(ObjectSet, Boolean),
    lambda x:{**x,
    "end":torch.max(x["end"], dim = -1).values})

operator_forall = Primitive(
    "forall",
    arrow(ObjectSet, Boolean),
    lambda x:{**x,
    "end":torch.min(x["end"], dim = -1).values})
