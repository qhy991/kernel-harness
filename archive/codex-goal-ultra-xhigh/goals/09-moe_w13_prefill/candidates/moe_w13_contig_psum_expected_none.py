"""PSUM W13 control with no expected-M heuristic."""

from serving_native.candidates._moe_w13_contig_psum import prepare_psum, run_psum


METADATA = {
    "control": "expected_m_none",
    "ensure_zero_padding": False,
    "expected_m_for_psum_layout": None,
    "compiled_dims": "nk",
}


def prepare(inputs, runtime):
    return prepare_psum(inputs, runtime, METADATA)


def run(inputs, runtime):
    return run_psum(inputs, runtime, METADATA)
