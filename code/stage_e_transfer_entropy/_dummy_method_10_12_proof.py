"""
Section 10.12 — Extensibility Proof: synthetic 5th-method data generator.

**Not a pipeline stage.** Never imported by, and makes no changes to,
``11_symbolic_ete_core.py``, ``12_asymmetry_index_core.py``, or
``16_cross_method_significance.py``. This script exists solely to
manufacture on-disk data for a synthetic ``dummy_test_method`` registry
entry (see the commented-out block in ``config.yaml``'s
``significance_methods``) so that registering a 5th connectivity method can
be proven to require *only*:

  1. Generating/obtaining a ``B_{Exp,Nov}_*.npz`` mean-array pair with the
     standard ``(n_subs, 20, n_trials)`` schema (this script), and
  2. One new ``significance_methods`` entry with
     ``ai_loader.kind: "symbolic_mean_npz"`` (already-existing, generic
     loader kind — no new Python needed at all).

Data here is a **trivial reshuffle** of Phase 4's real raw amplitude-dSTE
subject-mean arrays (connection axis permuted per group, then
absolute-valued so the same AI-boundedness assumptions from Section 10.10's
raw-magnitude tracks hold) — deliberately not random noise, so the pipeline
exercises real-shaped, non-degenerate data rather than a trivial all-zero
edge case.

Usage (temporary, proof-only)
------------------------------
1. Uncomment the ``dummy_test_method`` block in ``config.yaml``.
2. ``python3 code/stage_e_transfer_entropy/_dummy_method_10_12_proof.py``
3. ``python3 code/stage_e_transfer_entropy/12_asymmetry_index_core.py --all --overwrite``
4. ``python3 code/stage_e_transfer_entropy/16_cross_method_significance.py --overwrite``
5. Confirm ``dummy_test_method`` appears in both tools' logs/outputs.
6. Re-comment the registry entry, delete this script's output directory
   (``outputs/.../_dummy_method_10_12_proof/``), and re-run steps 3-4 to
   restore the 4 real tracks' outputs (unaffected either way, since this
   script never touches their files).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CODE_ROOT = Path(__file__).parent.parent
_STAGE_E_ROOT = Path(__file__).parent

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))
if str(_STAGE_E_ROOT) not in sys.path:
    sys.path.insert(0, str(_STAGE_E_ROOT))

from utils import load_config  # noqa: E402


def generate_dummy_arrays(config: dict | None = None, seed: int = 999) -> tuple[np.ndarray, np.ndarray]:
    from utils_te import get_te_output_dir

    if config is None:
        config = load_config()

    src_dir = get_te_output_dir("phase_4_group", config=config)
    B_Exp = np.load(src_dir / "B_Exp_amplitude_mean.npz", allow_pickle=True)["data"]
    B_Nov = np.load(src_dir / "B_Nov_amplitude_mean.npz", allow_pickle=True)["data"]

    rng = np.random.default_rng(seed)
    B_Exp_dummy = np.abs(B_Exp[:, rng.permutation(B_Exp.shape[1]), :])
    B_Nov_dummy = np.abs(B_Nov[:, rng.permutation(B_Nov.shape[1]), :])

    out_dir = get_te_output_dir("dummy_method_group", config=config)
    np.savez(out_dir / "B_Exp_dummy_mean.npz", data=B_Exp_dummy)
    np.savez(out_dir / "B_Nov_dummy_mean.npz", data=B_Nov_dummy)
    print(f"generate_dummy_arrays: saved B_Exp{B_Exp_dummy.shape} / B_Nov{B_Nov_dummy.shape} -> {out_dir}")
    return B_Exp_dummy, B_Nov_dummy


if __name__ == "__main__":
    generate_dummy_arrays()
