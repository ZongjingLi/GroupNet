from mvcl.config import config
from mvcl.model import MetaVisualLearner
from mvcl.primitives import *
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

# [Demo Model]
domain = load_domain_string(meta_domain_str, domain_parser)
model = MetaVisualLearner(domain, config)

context = {
        "end":logit(torch.tensor([0,1,1,1,0])),
        "masks": logit(torch.ones(5,32*32)),
        "features": torch.randn([32*32,100]),
        "model": model
}

p = Program.parse("(count (scene $0))")
p = Program.parse("( && (not(exists (scene $0))) (exists (scene $0)) )")
p = Program.parse("(filter(forall (expand (scene $0))) (scene $0))")
p = Program.parse(f"""
    (exists
        (intersect
            (exists
                (Pr(expand (scene $0)) is-red ) 
            )
            (forall
                (Pr (expand(scene $0))  is-ship)
            )
        )
    )
    """)
#p = Program.parse("(expand (scene $0))")

o = p.evaluate({0:context})
print(o["end"])

