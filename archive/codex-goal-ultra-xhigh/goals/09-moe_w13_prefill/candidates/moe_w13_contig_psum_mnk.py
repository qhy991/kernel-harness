"""PSUM W13 control compiling M, N, and K instead of N and K only."""

from serving_native.candidates._moe_w13_contig_psum import prepare_psum, run_psum


METADATA = {
    "control": "compiled_dims_mnk",
    "ensure_zero_padding": False,
    "expected_m_for_psum_layout": 1024,
    "compiled_dims": "mnk",
}


def prepare(inputs, runtime):
    return prepare_psum(inputs, runtime, METADATA)


def run(inputs, runtime):
    return run_psum(inputs, runtime, METADATA)
