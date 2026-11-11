"""What a model family **requires**, stated in semantic terms, with a mandatory/optional split.

The family table answers "which vLLM name do I take over".  That is the wrong question to ask when
something is missing, because it says nothing about *consequences*: today a failed takeover is a
warning, the service starts, and the first GDN prefill or MoE forward fails three layers down with a
stack that no longer mentions the port.

This module is the other half: one requirement per semantic name, whether the family can run without
it, and why.  Two consumers:

* ``ops/install.py`` records ``report.missing_mandatory`` / ``report.missing_optional`` so the banner
  and ``doctor`` can say "this configuration cannot serve a GDN model" instead of "10 bound";
* ``doctor`` turns a missing **mandatory** op into a failed check, and a missing optional one into a
  named warning.

Why "mandatory" is per op and not per family: the GDN family's ``chunk_local_cumsum`` is *declined*
on purpose (our port is not FLA's drop-in, and the ported chunked rule computes it internally), while
``chunk_gated_delta_rule`` is the whole point of the family.  A family-level verdict cannot express
that difference, and a boolean "supported" would force a lie in one direction.

Nothing here names vLLM: the engine path for each semantic name is ``engine/abi``'s business.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

__all__ = [
    "Requirement",
    "FAMILY_REQUIREMENTS",
    "requirements",
    "mandatory",
    "optional",
    "classify",
    "unknown_requirements",
]


@dataclass(frozen=True)
class Requirement:
    """One semantic op a family needs, and what happens without it."""

    semantic: str
    family: str
    mandatory: bool
    why: str = ""

    def __post_init__(self) -> None:
        if "." not in self.semantic:
            raise ValueError(f"requirement {self.semantic!r} must be 'family.operation'")


#: The GDN family, which the target models cannot serve without: every prefill and decode step runs
#: through these.  ``chunk_local_cumsum`` is optional because the ported chunked delta rule computes
#: the in-chunk cumulative sum itself (the FLA entry point is a different signature - see the
#: ``abi_ok=False`` record), so the op being declined costs nothing at run time.
_GDN = (
    Requirement("gdn.chunk_gated_delta_rule", "gdn", True, "the chunked recurrence itself"),
    Requirement("gdn.l2norm", "gdn", True, "q/k normalisation on every prefill"),
    Requirement("gdn.fused_sigmoid_gating", "gdn", True, "decode gating update"),
    Requirement("gdn.recurrent_decode", "gdn", True, "decode recurrence (packed)"),
    Requirement("gdn.post_conv_prep", "gdn", True, "prefill packing after the conv"),
    Requirement("gdn.conv1d_prefill", "gdn", True, "causal conv over a prefill chunk"),
    Requirement("gdn.conv1d_update", "gdn", True, "causal conv state update"),
    Requirement(
        "gdn.chunk_local_cumsum",
        "gdn",
        False,
        "the ported chunked rule computes it internally; the FLA entry point is not a drop-in",
    ),
)

#: The MoE family.  ``swiglu_gate`` is optional because the dense/MLP path only uses it when the
#: ``dense`` switch is on; ``experts`` is mandatory because a routed-expert model cannot forward
#: without it.
_MOE = (
    Requirement("moe.routing_topk_softmax", "moe", True, "router top-k + softmax"),
    Requirement("moe.align_block_size", "moe", True, "token-to-expert grouping"),
    Requirement("moe.experts", "moe", True, "the routed expert MLP"),
    Requirement("engine.moe_method", "moe", True, "the class vLLM instantiates for the OOT path"),
    Requirement("moe.swiglu_gate", "moe", False, "only used by the opt-in dense family"),
)

FAMILY_REQUIREMENTS: Mapping[str, Tuple[Requirement, ...]] = {"gdn": _GDN, "moe": _MOE}


def requirements(family: str) -> Tuple[Requirement, ...]:
    """Every requirement declared for ``family``; empty for a family that declares none."""
    return FAMILY_REQUIREMENTS.get(family, ())


def mandatory(family: str) -> Tuple[str, ...]:
    """The semantic names ``family`` cannot run without."""
    return tuple(r.semantic for r in requirements(family) if r.mandatory)


def optional(family: str) -> Tuple[str, ...]:
    """The semantic names ``family`` uses when present and does without when not."""
    return tuple(r.semantic for r in requirements(family) if not r.mandatory)


def classify(bound: Iterable[str], families: Mapping[str, bool]) -> Tuple[List[str], List[str]]:
    """``(missing_mandatory, missing_optional)`` given what bound and which families are on.

    ``bound`` holds *semantic* names (``report.bound_semantics``), so this stays a pure function over
    names - the caller owns the translation from vLLM-facing names.
    """
    have = set(bound)
    missing_mandatory: List[str] = []
    missing_optional: List[str] = []
    for family, enabled in families.items():
        if not enabled:
            continue
        for requirement in requirements(family):
            if requirement.semantic in have:
                continue
            target = missing_mandatory if requirement.mandatory else missing_optional
            target.append(requirement.semantic)
    return sorted(missing_mandatory), sorted(missing_optional)


def unknown_requirements(names: Sequence[str]) -> Tuple[str, ...]:
    """Requirements whose semantic names are not in ``contracts/vocabulary.py`` (should be empty)."""
    from ascend_platform.contracts.vocabulary import validate

    return validate(names)
