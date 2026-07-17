"""Aggregates every family's specs. Add a family = import it here."""
from .families import gemm, norm, elementwise, moe_gate, attention, dsa

FAMILIES = [gemm, norm, elementwise, moe_gate, attention, dsa]


def all_specs():
    out = []
    for mod in FAMILIES:
        out.extend(mod.specs())
    return out
