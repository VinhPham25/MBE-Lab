"""
Stage E — Transfer Entropy Effective Connectivity Pipeline

Implements a multi-phase Transfer Entropy (TE) pipeline that produces directed
connectivity matrices in the same format as Stage D (Linear GC), enabling direct
methodological comparison:

  Phase 1 — Bivariate dSPTE:     Delay Symbolic Phase Transfer Entropy (Wang 2020)
  Phase 2 — Multivariate dSPTE:  Phase 1 + multivariate conditioning (Wang 2022 inspired)
  Phase 3 — Bimodal dSPTE:       Phase + Amplitude branch fusion
  Phase 4 — Amplitude dSTE:      Multivariate conditional amplitude-only TE (current primary method)

Reference papers:
  Wang (2020): doi:10.1088/1741-2552/abb4a4
  Wang (2022): doi:10.1016/j.bspc.2021.103422
"""
