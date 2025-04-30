import torch
from loguru import logger
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
    logger.info("tiberate_compiler() called with FX graph:")
    # gm.graph.print_tabular()
    return gm.forward


aten = torch.ops.aten
default_decompositions = {aten.addmm}
# Reference https://pytorch.org/docs/stable/torch.compiler_custom_backends.html
tiberate_compiler = aot_autograd(
    fw_compiler=_compiler,
    decompositions=get_decompositions(default_decompositions),
)
