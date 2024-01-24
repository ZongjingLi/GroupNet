'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-01-24 17:50:12
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-01-24 18:24:56
 # @ Description: This file is distributed under the MIT license.
'''

from mvcl.model import MetaNet
from mvcl.train import train

from rinarak.domain import Domain, load_domain_string

domain_parser = Domain("mvcl/base.grammar")
meta_domain_str = f"""
(domain meta_demo)
(:type
    object - vector[float,100]
    position - vector[float,2]
    color - vector[float, 64]
    category
)
(:predicate
    color ?x-object -> vector[float,64]
    is-red ?x-object -> boolean
    is-blue ?x-object -> boolean
    is-ship ?x-object -> boolean
    is-house ?x-object -> boolean
)
(:derived
    is-green ?x-color expr: (??f ?x)
)
(:constraint
    (color: is-red is-blue)
    (category: is-ship is-house)
)
"""
domain = load_domain_string(meta_domain_str, domain_parser)
domain.print_summary()