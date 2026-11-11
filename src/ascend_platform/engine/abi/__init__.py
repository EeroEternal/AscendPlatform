"""Where the **engine's** ABI lives: what vLLM 0.23 calls things, and how its classes are extended.

This package is the only place in the platform that is allowed to know a vLLM module path, a vLLM
class name, or the shape of a vLLM extension point.  Three reasons it is a package of its own rather
than a corner of ``ops/``:

1. **It is versioned knowledge.**  ``vllm_0_23`` names a point release.  When 0.24 moves
   ``fused_moe.unquantized_fused_moe_method``, a new module is added next to this one and the
   supported range decides which is loaded - instead of a symbol string buried in a family table
   that nothing dates.
2. **It is the direction of dependency.**  ``engine -> ops -> contracts``.  The operator layer must
   never import this package (``tests/test_layer_acl.py`` proves it), so a semantic name in a family
   table stays engine-free and the mapping to ``vllm.*`` is applied *here*, or injected from here.
3. **The shapes differ from "a function".**  ``CustomOp.register_oot`` wants ``base_cls -> subclass``;
   the routed-expert path wants an object with ``is_monolithic`` as an attribute and
   ``experts_cls.is_monolithic()`` as a method.  Those asymmetries are ABI facts, and they belong
   next to the version that has them.
"""
