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

operator_equal_concept = Primitive(
    "equal_concept",
    arrow(ObjectSet, ObjectSet, Boolean),
    "Not Implemented"
)

# make reference to the objects in the scene
operator_related_concept = Primitive(
    "relate",
    arrow(ObjectSet, ObjectSet, Boolean),
    "Not Implemented"
)

def type_filter(objset,concept,exec):
    filter_logits = torch.zeros_like(objset["end"])
    parent_type = exec.get_type(concept)
    for candidate in exec.type_constraints[parent_type]:
        filter_logits += exec.entailment(objset["features"],
            exec.get_concept_embedding(candidate)).sigmoid()

    div = exec.entailment(objset["features"],
            exec.get_concept_embedding(concept)).sigmoid()

    filter_logits = logit(div / filter_logits)

    return torch.min(objset["end"],filter_logits)

def refractor(exe, name):
    exe.redefine_predicate(
        name,
        lambda x: {**x, "end":  type_filter(x, name, x["executor"]) }
    )

# end points to train the clustering methods using uniform same or different.
operator_uniform_attribute = Primitive("uniform_attribute",
                                       arrow(ObjectSet, Boolean),
                                       "not")

operator_equal_attribute = Primitive("equal_attribute",
                                     arrow(ObjectSet, Boolean, Boolean),
                                     "not"
                                     )