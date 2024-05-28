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


infinity = 1e9
device = "cuda:0" if torch.cuda.is_available() else "cpu"
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
    if concept in objset["context"]: return torch.min(objset["context"][concept], objset["end"])

    filter_logits = torch.zeros_like(objset["end"])
    parent_type = exec.get_type(concept)
    for candidate in exec.type_constraints[parent_type]:
        filter_logits += exec.entailment(objset["context"]["features"],
            exec.get_concept_embedding(candidate)).sigmoid()

    div = exec.entailment(objset["context"]["features"],
            exec.get_concept_embedding(concept)).sigmoid()

    filter_logits = logit(div / filter_logits)
   
    return torch.min(objset["end"],filter_logits)

def refractor(exe, name):
    exe.redefine_predicate(
        name,
        lambda x: {**x,"from":name, "set":x["end"], "end": type_filter(x, name, x["context"]["executor"]) }
    )

# end points to train the clustering methods using uniform same or different.
operator_uniform_attribute = Primitive("uniform_attribute",
                                       arrow(ObjectSet, Boolean),
                                       "not")

operator_equal_attribute = Primitive("equal_attribute",
                                     arrow(ObjectSet, Boolean, Boolean),
                                     "not"
                                     )

def condition_assign(x, y):
    """evaluate the expression x, y and return the end as an assignment operation"""
    return {**x, "end": [{"x": x["set"],"y": y["set"], "v" : y["end"], "c": torch.tensor(infinity, device = device), "to": x["from"]}]}

operator_assign_attribute = Primitive("assign",arrow(Boolean, Boolean, Boolean),
                                      lambda x: lambda y: condition_assign(x, y))

def condition_if(x, y):
    """x as the boolean expression to evaluate, y as the code blocks"""
    outputs = []
    for code in y["end"]:
        code_condition = code["c"] if isinstance(code["c"], torch.Tensor) else torch.tensor(code["c"])
        assign_operation = {
            "x": code["x"],
            "y": code["y"],
            "v": code["v"],
            "c": torch.min(code_condition, torch.max(x["end"])),
            "to": code["to"],
        }
        outputs.append(assign_operation)
    return {**x, **y, "end":outputs}

operator_if_condition = Primitive("if", arrow(Boolean, Boolean, Boolean),
                                  lambda x: lambda y: condition_if(x, y))

operator_pi = Primitive("pi", arrow(Boolean), {"end":torch.tensor(3.14), "set":torch.tensor(1.0)})

operator_true = Primitive("true", arrow(Boolean), {"end":torch.tensor(14.), "set":torch.tensor(1.0)})

operator_true = Primitive("false", arrow(Boolean), {"end":-1. * torch.tensor(14.), "set":torch.tensor(1.0)})