# IDTxl Evaluation Log

Generated: 2026-04-19 (Section 7.4.1)

## Status: AVAILABLE ✓

### Import Result

```
from idtxl.multivariate_te import MultivariateTE — SUCCESS
from idtxl.data import Data — SUCCESS
```

### Version
idtxl 1.6 (editable install)

### Installation
IDTxl was installed from https://github.com/pwollstadt/IDTxl via:
```bash
git clone https://github.com/pwollstadt/IDTxl
cd IDTxl
pip install -e .
```

Install confirmed by `pip show idtxl` in the `.venv` environment — all dependencies
(networkx, numpy, scipy, etc.) present in the project virtual environment.

### Estimator Used
`JidtKraskovCMI` — KSG conditional mutual information estimator operating on
continuous data (amplitude values after 0.01–0.10 Hz narrowband filtering).
Bypasses discrete binning and the contingency-table sparsity problem that limits
Phase 2 NG-symbolic conditioning.

### Why IDTxl (not NG-symbolic) as validation reference
IDTxl's `MultivariateTE` implements multivariate TE with non-uniform embedding
(Faes 2011 NUE method) using the KSG k-NN estimator — this is the closest
continuous-domain analogue to what Granger Causality does.  Key differences:

| Property | Phase 2 NG-Symbolic | IDTxl MultivariateTE |
|---|---|---|
| Estimator | Frequency counting (discrete) | KSG k-NN (continuous) |
| Embedding | Uniform (fixed τ, d=2) | Non-uniform (NUE, greedy selection) |
| Sparsity | 4D contingency table, ~16 cells | No binning — no sparsity |
| Input | Hilbert phase (symbolic) | Raw amplitude (continuous) |
| Significance | Not tested per connection | Built-in permutation test |
| Speed (per window) | ~0.2 s | ~30–120 s (KSG + permutations) |

### Settings for ASTaUND fNIRS Data

| Parameter | Value | Rationale |
|---|---|---|
| `cmi_estimator` | `JidtKraskovCMI` | Continuous KSG — no binning |
| `max_lag_sources` | `u_max` (default 50) | Covers ~2 s hemodynamic lag |
| `min_lag_sources` | 1 | Allow all lags ≥ 1 sample |
| `max_lag_target` | 20 | Target history: shorter for speed |
| `n_perm_max_stat` | 50 | Reduced for speed on fNIRS windows |
| `n_perm_min_stat` | 50 | Reduced for speed |
| `n_perm_omnibus` | 100 | Omnibus permutation test |
| `n_perm_max_seq` | 50 | Reduced for speed |
| `alpha_*` | 0.05 | Standard significance threshold |
| `normalise` | True | Z-standardise per ROI (IDTxl default) |

### Comparison Design (Section 7.4.3)
- IDTxl run on the same pre-filtered (0.01–0.10 Hz) amplitude windows as Phase 2
- IDTxl uses raw continuous amplitude (not symbolic sequences)
- Output: 5×5 EC matrix per window → 20-element connectivity vector
  using same `get_connection_labels()` schema as GC and Phase 2
- Pearson r between Phase 2 NG-symbolic and IDTxl vectors, averaged across
  subjects × windows, is the primary comparison statistic

### Interpretation Thresholds (from PRD 7.4.3)

| Pearson r | Interpretation | Action |
|---|---|---|
| r > 0.6 | NG-symbolic Phase 2 validated | Report NG-symbolic as primary |
| 0.3 < r ≤ 0.6 | Partial agreement | Report both; NG-symbolic methodologically faithful |
| r ≤ 0.3 | Substantial disagreement | Elevate IDTxl to primary Phase 2 method |

### Computational Cost Warning
IDTxl MultivariateTE with permutation testing is substantially slower than
NG-symbolic:

- Phase 2 NG-symbolic: ~0.2 s per window
- IDTxl (n_perm=50, T=1351, 5 ROIs): estimated **30–120 s per window**

For the full batch (13 subjects × ~4 trials × ~2 windows per trial):
~13 × 4 × 2 × 75 s ≈ **52 minutes** (optimistic estimate).

Use `--first-window-only` flag for rapid validation (1 window per trial).

### Implementation
Implemented in `code/stage_e_transfer_entropy/10_idtxl_validation.py`:

- `write_idtxl_eval_md()` — Section 7.4.1
- `run_idtxl_multivariate_te()` — Section 7.4.2
- `extract_idtxl_connectivity_vector()` — 5×5 → 20-vector helper
- `run_subject_idtxl()` — per-subject runner
- `run_batch_idtxl()` — batch runner for all 13 subjects
- `compare_symbolic_vs_idtxl()` — Section 7.4.3 scatter + JSON

### Output Files
- `phase_2_multivariate_dspte/idtxl_validation/sub-{id}_idtxl_ec.npz` — per-subject
- `phase_2_multivariate_dspte/idtxl_validation/idtxl_batch_log.json` — batch log
- `phase_2_multivariate_dspte/idtxl_validation/idtxl_comparison_summary.json` — comparison
- `phase_2_multivariate_dspte/visualizations/symbolic_vs_idtxl_scatter.png` — scatter plot
