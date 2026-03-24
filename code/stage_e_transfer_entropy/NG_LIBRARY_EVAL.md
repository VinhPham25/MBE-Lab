# Neural-Gas PyPI Library Evaluation

**Date:** 2026-03-23  
**Task:** 1.1.2 — Evaluate PyPI Neural-Gas libraries against checklist before committing to implementation strategy.  
**Decision:** Implement from scratch (see conclusion below).

---

## Checklist Criteria

Per `TRANSFER_ENTROPY_PROGRESS.md` §1.1.2, a suitable PyPI library must satisfy:

| Criterion | Requirement |
|-----------|-------------|
| (a) | Accepts a 2D numpy array as input |
| (b) | Exposes `n_neurons` / `k` as a configurable hyperparameter |
| (c) | Allows iterative fitting OR exposes distortion error per iteration |
| (d) | Returns prototype vectors and/or assignments |

---

## Libraries Evaluated

### 1. `neuralgas` (PyPI)
- **Search result:** `pip install neuralgas` → `ERROR: Could not find a version that satisfies the requirement neuralgas (from versions: none)`.
- **Status:** Package does not exist on PyPI as of 2026-03-23.
- **Verdict:** ❌ Not available.

### 2. `neural-gas` (PyPI)
- **Search result:** `pip install neural-gas` → `ERROR: Could not find a version that satisfies the requirement neural-gas (from versions: none)`.
- **Status:** Package does not exist on PyPI as of 2026-03-23.
- **Verdict:** ❌ Not available.

### 3. `sklearn` / `MiniBatchKMeans` (as fallback quantiser)
- K-means is a competitive learning algorithm like Neural-Gas, but it uses hard assignment (winner-take-all) rather than Neural-Gas's soft-neighbourhood ranking. K-means does NOT satisfy Wang (2020)'s NG-specific distortion metric (nDis based on the NG codebook). The soft-ranking update rule of NG produces different prototype placement than k-means, particularly near cluster boundaries — which affects symbolic sequence quality.
- **Verdict:** ❌ Does not replicate NG methodologically; not suitable as drop-in.

### 4. `somoclu` (Self-Organising Maps — related family)
- SOMs use a grid-structured neighbourhood; NG uses a rank-based neighbourhood with no fixed topology. Different algorithms with different prototype placement properties.
- **Verdict:** ❌ Not methodologically equivalent to Neural-Gas.

---

## Decision: Scratch Implementation

**Rationale:** No suitable PyPI library exists. The Neural-Gas algorithm is approximately 50 lines of code and is well-specified in Martinetz et al. (1993). The scratch implementation:

1. Fully controls the nDis stopping criterion (required for Wang 2020 §3.2)
2. Returns prototype vectors and full assignment arrays
3. Is configurable: `k`, `max_iterations`, `λ_i`, `λ_f`, `σ_i`, `σ_f`
4. Can be independently validated on Wang (2020)'s AR simulation (N=20,000 samples, δ=3, a=0.9)

**Implementation location:** `code/stage_e_transfer_entropy/utils_te.py` — class `NeuralGas`.

---

## Neural-Gas Algorithm Reference

**Source:** Martinetz, T.M., Berkovich, S.G., & Schulten, K.J. (1993). "Neural-gas" network for vector quantization and its application to time-series prediction. *IEEE Trans. Neural Netw.*, 4(4), 558–569.

**Algorithm sketch (for reference):**

```
Inputs:  X ∈ R^(N×d),  k (n_prototypes),  T_max,  λ_i, λ_f,  σ_i, σ_f
Outputs: W ∈ R^(k×d)   (prototype vectors)

Initialize: W = random sample of k rows from X

For t = 0 to T_max - 1:
    ε  = λ_i * (λ_f / λ_i)^(t / T_max)    # decaying learning rate
    σ  = σ_i * (σ_f / σ_i)^(t / T_max)    # decaying neighbourhood width

    Pick random x from X
    Compute d_i = ‖x - W_i‖²  for all i = 1..k
    Sort prototypes by distance: rank k_i (0 = nearest)
    Update: W_i ← W_i + ε * h(k_i, σ) * (x - W_i)
            where h(k_i, σ) = exp(-k_i / σ)  (soft-neighbourhood kernel)

nDis = Σ ‖x_t - W_{nearest(t)}‖² / Σ ‖x_t - mean(X)‖²
```

The nDis criterion (Wang 2020 Eq. 11) is evaluated after convergence at each k. The search stops at the smallest k for which nDis < 0.05.
