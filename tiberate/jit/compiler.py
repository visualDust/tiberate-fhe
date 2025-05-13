import os

import torch
from torch._decomp import get_decompositions
from torch._dynamo import register_backend
from torch._dynamo.backends.common import aot_autograd

# from torch._inductor.fx_passes.dedupe_symint_uses import dedupe_symints
# from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops


def _process_fx(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    # some passes on the FX graph
    # dedupe_symints(gm.graph)
    # reinplace_inplaceable_ops(gm.graph)
    return gm


@register_backend
def _compiler(gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
    gm = _process_fx(gm, example_inputs)

    if os.getenv("DEBUG_MODE") == "1":
        gm.graph.print_tabular()
        import time

        import ubelt as ub
        from torch.fx import passes

        random_name = str(int(time.time() * 1000))
        g = passes.graph_drawer.FxGraphDrawer(gm, "tiberate_compiler")
        dpath = (
            ub.Path(__file__).parent.parent.parent / "tests" / "FxGraphDrawer"
        ).ensuredir()
        fpath = dpath / f"{random_name}.svg"
        g.get_dot_graph().write_svg(fpath)
    return gm.forward


aten = torch.ops.aten
default_decompositions = {aten.addmm}
# Reference https://pytorch.org/docs/stable/torch.compiler_custom_backends.html
tiberate_compiler = aot_autograd(
    fw_compiler=_compiler,
    decompositions=get_decompositions(default_decompositions),
)
