"""Diagnostic PSUM W2 variant specializing the replay's total aligned M."""

from serving_native.candidates._moe_w2_contig_psum import prepare_psum, run_psum


METADATA = {
    "control": "use_psum_layout",
    "ensure_zero_padding": False,
    "expected_m_for_psum_layout": 1024,
    "compiled_dims": "mnk",
    "promotion_risk": "dynamic routing changes total M; diagnostic only",
}



def prepare(inputs, runtime):
    return prepare_psum(inputs, runtime, METADATA)


def run(inputs, runtime):
    return run_psum(inputs, runtime, METADATA)
