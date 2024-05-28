from rinarak.domain import load_domain_string, Domain
from rinarak.knowledge.executor import CentralExecutor
domain_parser = Domain("mvcl/base.grammar")
from mvcl.primitives import *
from dataclasses import dataclass
import sys

meta_domain_str = ""
with open(f"domains/demo_domain.txt","r") as domain:
        for line in domain: meta_domain_str += line
domain = load_domain_string(meta_domain_str, domain_parser)


executor = CentralExecutor(domain, "cone", 100)
for predicate in executor.predicates[1]:
      refractor(executor, predicate.name)

rand_end = torch.ones([4])
rand_feat = torch.randn([4,100])

"""

optimizer = torch.optim.Adam(executor.parameters(), lr = 1e-1)
for epoch in range(100):
    context = {
        "end": logit(rand_end),
        "pos": torch.randn([4,3]),
        "red": torch.randn([4]), #{"end": torch.ones([4])},,
        "features": rand_feat,
        "left": torch.ones([4,4]),
        "executor": executor#{"end": torch.zeros([4])}
    }

    precond, effect_output = executor.apply_action("spreader", params = [1,3], context = context)
    
    sys.stdout.write(f"\rprecond:{precond.detach().numpy()}")
    loss =torch.nn.functional.mse_loss(effect_output["red"][1], torch.tensor(3.14))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""
print()




#precond, effect_output = executor.apply_action("hold", params = [1,3], context = context)
#print(precond.detach().numpy(),effect_output["red"].detach().numpy())

from rinarak.algs.search import run_heuristic_search
import random

context = {
        "end": logit(rand_end),
        "pos": torch.randn([4,3]),
        "red": -8*torch.ones([4]), #{"end": torch.ones([4])},,
        "green": -8*torch.ones([4]),
        "features": rand_feat,
        "left": torch.ones([4,4]),
        "executor": executor#{"end": torch.zeros([4])}
    }


"""
print(context["red"].cpu().detach().numpy())
print(context["green"].cpu().detach().numpy())

precond, effect_output = executor.apply_action("make_green", params = [0], context = context)
print("make green->")
print(effect_output["red"])
print(effect_output["green"])


precond, effect_output = executor.apply_action("spreader", params = [1,3], context = effect_output)
print("spearder->")
print(effect_output["red"])
print(effect_output["green"])
"""



print("start the goal search")
print(context["red"].cpu().detach().numpy())
print(context["green"].cpu().detach().numpy())
state, action, costs, nr = executor.search_discrete_state(context, "(exists (red $0) )")
print(state[-1].state["red"].cpu().detach().numpy())
print(state[-1].state["green"].cpu().detach().numpy())
print()
print(action)
print(costs)
print(nr)
