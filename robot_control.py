from rinarak.domain import load_domain_string, Domain
from rinarak.knowledge.executor import CentralExecutor
domain_parser = Domain("mvcl/base.grammar")
from rinarak.dsl.vqa_primitives import *

meta_domain_str = ""
with open(f"domains/demo_domain.txt","r") as domain:
        for line in domain: meta_domain_str += line
domain = load_domain_string(meta_domain_str, domain_parser)


executor = CentralExecutor(domain, "cone", 100)

context = {
        "end": torch.randn([4,4]), #{"end": torch.ones([4])},,
        "executor": executor#{"end": torch.zeros([4])}
}

Primitive.GLOBALS["exists"].value = lambda x:{**x,"end":torch.max(x["end"] ,dim = -1).values}

executor.apply_action("put", params = [0,2], context = context)