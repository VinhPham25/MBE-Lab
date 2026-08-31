# Pipeline outputs

This directory is gitignored (regenerated `.npz` / `.png` / `.csv`). The
naming convention, folder map, and a catalog of what each file **shows**
are tracked here:

**[docs/OUTPUTS.md](../docs/OUTPUTS.md)**

## Filename grammar (figures)

```
{scope}_{measure}_{view}[_{facet}][_Tr{N}][_Win{M}].{ext}
```

Example: `group_te_bar_expert_vs_novice.png`

Do **not** prefix with `p4_` / `phase_4_` / `expert-novice` — the parent
folders already say that. `expert_vs_novice` as a *comparison type* is fine.

Data files (`.npz`, the four H1/H0/HIV/normality CSVs) keep their existing
schema — see `docs/OUTPUTS.md` §2.4.
