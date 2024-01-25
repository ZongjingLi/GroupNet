'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-01-25 04:37:41
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-01-25 04:37:42
 # @ Description: This file is distributed under the MIT license.
'''
from re import X
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


def expand_like(source, target):
    assert source.size(0) == target.size(0)
    expand_mults = [1]
    for i in range(len(target.shape) - 1):
        source = source.unsqueeze(-1)
        expand_mults.append(target.shape[1 + i])
    source = source.repeat(expand_mults)
    return source

# [Return all the objects in the Scene]
operator_scene = Primitive(
    "scene",
    arrow(Stream, ObjectSet),
    lambda x: {"end": x["end"], **x}
)

# [Existianial quantification, exists, forall]
operator_exists = Primitive(
    "exists",
    arrow(ObjectSet, Boolean),
    lambda x:{"end":torch.max(x["end"], dim = -1).values, "model":x["model"]})
operator_forall = Primitive(
    "forall",
    arrow(ObjectSet, Boolean),
    lambda x:{"end":torch.min(x["end"], dim = -1).values, "model":x["model"]})

# [Boolean operators]
operator_and = Primitive(
    "&&",
    arrow(Boolean, Boolean, Boolean),
    lambda x:lambda y:{"end":torch.min(x["end"],y["end"]), "model":x["model"]})

operator_or = Primitive(
    "||",
    arrow(Boolean, Boolean, Boolean),
    lambda x:lambda y:{"end":torch.max(x["end"],y["end"]), "model":x["model"]})

operator_not = Primitive(
    "not",
    arrow(Boolean, Boolean),
    lambda x:{"end":-x["end"], "model":x["model"]})

# [Neuro-Sybmolic quantification]
operator_pr = Primitive("Pr", arrow(ObjectSet,Concept,ObjectSet),
    lambda x: lambda y:
    {"end":x["model"].entailment(x["features"],
    x["model"].get_concept_embedding(y)), "model":x["model"]})

def relate(x,y,z):
    mask = x["executor"].entailment(x["relations"],x["executor"].get_concept_embedding(z))
    N, N = mask.shape
    score_expand_mask = torch.min(expat(x["end"],0,N),expat(x["end"],1,N))
    new_logits = torch.min(mask, score_expand_mask)
    return {"end":new_logits, "executor":x["executor"]}
operator_relate = Primitive(
    "relate",
    arrow(ObjectSet, ObjectSet, Concept, ObjectSet),
    lambda x:lambda y: lambda z: relate(x,y,z))

# [Set operations] Intersect and Union operations
operator_intersect = Primitive(
    "&",
    arrow(ObjectSet, ObjectSet, ObjectSet),
    lambda x: lambda y: {"end":torch.min(x, y),"model": x["model"]})

operator_union = Primitive(
    "|",
    arrow(ObjectSet, ObjectSet, ObjectSet),
    lambda x: lambda y: {"end":torch.min(x, y),"model": x["model"]},)

operator_filter = Primitive(
    "filter",
    arrow(ObjectSet, ObjectSet, ObjectSet),
    lambda x: lambda y: {"end": torch.min(x["end"], y["end"]), **x})

def visual_group_segment(x):
    # input a mask of feature map, and use visual grouping for segmentation.
    scores, masks = x["model"].segment(x["end"], x["features"])
    return {"end":1,"masks":1,**x}
operator_filter_part = Primitive(
    "segment",
    arrow(PrimitiveSet,ObjectSet),
    visual_group_segment)

operator_expand = Primitive(
    "expand",
    arrow(ObjectSet,PrimitiveSet),
    lambda x: {"end": torch.min(x["masks"],expand_like(x["end"], x["masks"])), "model":x["model"]}
)

# [Count based questions the number of elements in the set]
operator_count = Primitive(
    "count",
    arrow(ObjectSet, Integer),
    lambda x:{"end":torch.sum(x["end"].sigmoid()), **x})

operator_equal = Primitive(
    "equal",arrow(treal, treal, Boolean),
    lambda x: lambda y:  {"end":8 * (.5 - (x - y).abs()), **x})

if __name__ == "__main__":
    context = {
        "end":logit(torch.tensor([0,1,1,1,0])),
        "masks": logit(torch.ones([5,32 * 32])),
        "features": torch.randn([32*32,132]),
        "model": None
    }

    """
    ObjectSet: M = WxH
        s_i: N
        m_i: NxM
        F_i: NxMxD
    """

    p = Program.parse("(count (scene $0))")
    p = Program.parse("( && (not(exists (scene $0))) (exists (scene $0)) )")
    p = Program.parse("(filter(forall (expand (scene $0))) (scene $0))")

    o = p.evaluate({0:context})
    print(o)
