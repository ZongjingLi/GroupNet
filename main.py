'''
 # @ Author: Zongjing Li
 # @ Create Time: 2024-01-24 17:50:12
 # @ Modified by: Zongjing Li
 # @ Modified time: 2024-01-24 18:24:56
 # @ Description: This file is distributed under the MIT license.
'''

from mvcl.model import MetaVisualLearner
from mvcl.train import train
from mvcl.config import config

from rinarak.domain import Domain, load_domain_string

domain_name = "demo"
domain_parser = Domain("mvcl/base.grammar")
meta_domain_str = ""
with open(f"domains/{domain_name}_domain.txt","r") as domain:
    for line in domain: meta_domain_str += line
domain = load_domain_string(meta_domain_str, domain_parser)
domain.print_summary()

model = MetaVisualLearner(domain, config)
