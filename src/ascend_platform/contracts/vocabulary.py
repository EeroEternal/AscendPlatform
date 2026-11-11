"""The **semantic vocabulary**: what a ``platform_name`` may be, and what shape of thing it names.

Why a vocabulary exists at all
------------------------------
Two sides name the same operation and neither name is the other's business:

* the **engine** calls it ``vllm._custom_ops.topk_softmax`` - a module path that changes whenever
  vLLM reorganises, and which an operator repo must never contain;
* an **operator package** calls it ``moe.routing_topk_softmax`` - a family-scoped name that survives
  an engine upgrade.

The descriptor's ``platform_name`` is the second kind, and this module is the closed set it may come
from.  Without a closed set, "semantic name" degrades into "whatever string the two repos agreed on
last week", which is how the first version of this contract ended up keying a *class factory*
binding on a name whose target was a plain compute function: the lookup worked, the value was the
wrong kind of object, and the failure would have surfaced at model instantiation.

Kinds, because "an implementation" is not one shape
---------------------------------------------------
``compute``
    ``callable(*tensors, **kwargs) -> tensor`` - what a kernel-level binding wants.
``class_factory``
    ``callable(base_cls) -> subclass`` - what ``CustomOp.register_oot`` wants.  A compute function
    bound here is a *silent* wrong branch, so :func:`requires_factory` exists to be checked at bind
    time rather than discovered at run time.

The engine's own symbol table is **not** here: it lives in ``engine/abi/`` because it is the layer
that knows which vLLM release is loaded.  This module only says which names and kinds are legal.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

__all__ = [
    "OpKind",
    "SemanticOp",
    "SEMANTIC_OPS",
    "BY_NAME",
    "kind_of",
    "requires_factory",
    "names_for_family",
    "validate",
]


class OpKind(str, enum.Enum):
    """The two shapes an implementation can have.  See the module docstring."""

    COMPUTE = "compute"
    CLASS_FACTORY = "class_factory"


@dataclass(frozen=True)
class SemanticOp:
    """One name in the vocabulary: which family it belongs to, and what shape it must have."""

    name: str
    family: str
    kind: OpKind = OpKind.COMPUTE
    #: One line a reader can check against the operator repo's own description.
    note: str = ""

    def __post_init__(self) -> None:
        if "." not in self.name:
            raise ValueError(
                f"semantic op name {self.name!r} must be 'family.operation' - the dot is what makes "
                "it obviously not an engine symbol"
            )


#: The closed set.  Every name the family table asks a provider for must appear here; the dotted
#: prefix is the family, so ``moe.experts`` is the routed-expert compute of the MoE family.
SEMANTIC_OPS: Tuple[SemanticOp, ...] = (
    # --- MoE ---------------------------------------------------------------------------------
    SemanticOp("moe.routing_topk_softmax", "moe", note="router: top-k + softmax over expert logits"),
    SemanticOp("moe.align_block_size", "moe", note="sort tokens by expert, pad to block size"),
    SemanticOp("moe.swiglu_gate", "moe", note="silu(x[:d]) * x[d:]"),
    SemanticOp("moe.experts", "moe", note="routed expert MLP, unquantised"),
    SemanticOp(
        "moe.experts_unquantized",
        "moe",
        note="same MLP with the plain (hidden, w1, w2, topk_weights, topk_ids) signature",
    ),
    SemanticOp("moe.slot_layout", "moe", note="group-list -> slot layout"),
    SemanticOp("moe.prenz_pack", "moe", note="pack ND into compact NZ"),
    # --- GDN / linear attention --------------------------------------------------------------
    SemanticOp("gdn.gating", "gdn", note="sigmoid gating update"),
    SemanticOp("gdn.conv_update", "gdn", note="causal conv state update"),
    SemanticOp("gdn.chunk_wu", "gdn", note="chunked WY representation"),
    SemanticOp("gdn.chunk_gated_delta_rule", "gdn", note="chunked delta-rule recurrence"),
    SemanticOp("gdn.chunk_local_cumsum", "gdn", note="in-chunk cumulative sum"),
    SemanticOp("gdn.l2norm", "gdn", note="L2 normalise the q/k head dim"),
    SemanticOp("gdn.fused_sigmoid_gating", "gdn", note="packed decode gating (FLA)"),
    SemanticOp("gdn.recurrent_decode", "gdn", note="packed decode recurrence (FLA)"),
    SemanticOp("gdn.post_conv_prep", "gdn", note="prefill post-conv packing (FLA)"),
    SemanticOp("gdn.conv1d_prefill", "gdn", note="causal conv over a prefill chunk"),
    SemanticOp("gdn.conv1d_update", "gdn", note="causal conv state update (decode)"),
    # --- attention --------------------------------------------------------------------------
    SemanticOp("attention.paged_prefill", "attention", note="paged prefill attention"),
    # --- sampling ---------------------------------------------------------------------------
    SemanticOp("sampling.topk_threshold", "sampling", note="top-k then threshold"),
    SemanticOp("sampling.greedy", "sampling", note="argmax token selection"),
    SemanticOp("sampling.top_p", "sampling", note="nucleus filter"),
    # --- the one kind that is not compute ----------------------------------------------------
    SemanticOp(
        "engine.moe_method",
        "moe",
        kind=OpKind.CLASS_FACTORY,
        note="builds the engine's UnquantizedFusedMoEMethod subclass around a compute function",
    ),
)

BY_NAME: Dict[str, SemanticOp] = {op.name: op for op in SEMANTIC_OPS}


def kind_of(name: str) -> Optional[OpKind]:
    """The declared kind, or ``None`` for a name outside the vocabulary."""
    op = BY_NAME.get(name)
    return None if op is None else op.kind


def requires_factory(name: str) -> bool:
    """True when binding this name to a plain compute function would be a wrong branch.

    Unknown names answer ``False``: this function is a guard for *declared* factory slots, not a
    whitelist - an operator package may serve names the platform has not learned yet.
    """
    return kind_of(name) is OpKind.CLASS_FACTORY


def names_for_family(family: str) -> Tuple[str, ...]:
    """Every semantic name declared for one family (``"moe"``, ``"gdn"``, ...), in table order."""
    return tuple(op.name for op in SEMANTIC_OPS if op.family == family)


def validate(names: Iterable[str]) -> Tuple[str, ...]:
    """Return the names that are *not* in the vocabulary (a test asserts this is empty)."""
    return tuple(name for name in names if name not in BY_NAME)
