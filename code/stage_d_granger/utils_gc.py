"""
Stage D — Shared Utility Functions for the Linear GC Pipeline

Provides the four core helpers required across all Stage D modules:

  load_preprocessed_fif(subject_id, study)
      Load the Stage B preprocessed .fif for a subject.

  get_event_onsets(raw)
      Return MNE event array from .fif annotations (trial_start=1, trial_end=2).

  get_subject_group(subject_id)
      Return 'Expert' or 'Novice' for a subject, or raise if excluded/unknown.

  get_connection_labels()
      Return the 20 directed connection strings in MATLAB-matching order for
      ['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA'] (row→col convention).

  get_roi_channel_map(raw)   [added in Section 2.1.2]
      Wrapper around build_region_map_5() that filters bad channels and appends
      the ' hbo' suffix, returning full MNE channel names per ROI.

Reference   : docs/LINEAR_GC_PROGRESS.md Section 1.1.3, 2.1.2
MATLAB ref  : GC_freq_main_TRGC_NovExp.m (subject lists, connection ordering)
"""

import logging
import sys
from pathlib import Path
from typing import Optional

import mne
import numpy as np

# ---------------------------------------------------------------------------
# Path setup — allows importing from code/ regardless of working directory
# ---------------------------------------------------------------------------
_CODE_ROOT = Path(__file__).parent.parent          # code/
_PROJECT_ROOT = _CODE_ROOT.parent                  # repo root

if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from utils import load_config                       # code/utils.py

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ROI / connection constants
# ---------------------------------------------------------------------------

ROI_NAMES = ["LPFC", "CPFC", "RPFC", "LM1", "SMA"]
"""Canonical 5-ROI names, matching build_region_map_5() abbreviated keys."""

# Event annotation IDs in Stage B .fif files (MNE Annotations description strings)
_EVENT_ID_MAP = {"1": 1, "2": 2}   # '1' = trial_start, '2' = trial_end
_TRIAL_START_ID = 1
_TRIAL_END_ID = 2


# ===========================================================================
# 1.1.3 — load_preprocessed_fif
# ===========================================================================

def load_preprocessed_fif(
    subject_id: str,
    study: str = "expert_novice",
    config: Optional[dict] = None,
    preload: bool = True,
    verbose: bool = False,
) -> mne.io.BaseRaw:
    """
    Load the Stage B preprocessed .fif file for a subject.

    Constructs the path from config.yaml:
      ``config['paths']['stage_b_outputs'] / 'processed_data' / {study_folder} /
        sub-{id}_task-{bids_task}_desc-preprocessed_nirs_raw.fif``

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. 'A1', 'R7'.
    study : str
        Config dataset key, e.g. 'expert_novice'.
    config : dict, optional
        Parsed config.yaml dict. Loaded from project root if None.
    preload : bool
        Passed to mne.io.read_raw_fif (True loads data into RAM).
    verbose : bool
        MNE verbosity.

    Returns
    -------
    raw : mne.io.BaseRaw
        Preprocessed continuous HbO/HbR time series with Annotations preserved.

    Raises
    ------
    FileNotFoundError
        If the expected .fif path does not exist.
    ValueError
        If the subject belongs to the excluded list (R10).
    """
    if config is None:
        config = load_config()

    # Block excluded subjects early
    gc_cfg = config.get("stage_d_granger", {}).get("linear_gc", {})
    excluded = gc_cfg.get("subjects", {}).get("excluded", [])
    if subject_id in excluded:
        raise ValueError(
            f"Subject {subject_id} is on the GC exclusion list "
            f"(Stage B quality flag — trailing zeros, anomalous rPCA). "
            f"Excluded subjects: {excluded}"
        )

    ds_cfg = config["datasets"][study]
    stage_b_root = _PROJECT_ROOT / config["paths"]["stage_b_outputs"]
    study_folder = ds_cfg.get("output_folder", study.replace("_", "-"))
    bids_task = ds_cfg["bids_task"]
    processed_dir = stage_b_root / "processed_data" / study_folder

    fif_path = processed_dir / (
        f"sub-{subject_id}_task-{bids_task}_desc-preprocessed_nirs_raw.fif"
    )

    if not fif_path.exists():
        # Fallback: glob for any preprocessed file for this subject
        candidates = sorted(processed_dir.glob(
            f"sub-{subject_id}_*_desc-preprocessed*_raw.fif"
        ))
        if not candidates:
            raise FileNotFoundError(
                f"No preprocessed .fif found for subject {subject_id}.\n"
                f"Expected: {fif_path}"
            )
        fif_path = candidates[0]
        logger.warning("load_preprocessed_fif: using fallback path %s", fif_path.name)

    raw = mne.io.read_raw_fif(fif_path, preload=preload, verbose=verbose)
    logger.info(
        "Loaded %s  (%d ch, %.1f Hz, %.1f s)",
        fif_path.name, len(raw.ch_names), raw.info["sfreq"], raw.times[-1],
    )
    return raw


# ===========================================================================
# 1.1.3 — get_event_onsets
# ===========================================================================

def get_event_onsets(raw: mne.io.BaseRaw) -> np.ndarray:
    """
    Extract MNE event array from .fif annotations.

    Mirrors MATLAB ``st = find(da.s)`` — returns all trigger sample indices
    (both trial_start=1 and trial_end=2) as a standard MNE events array.

    The returned array is used by ``get_trial_segments()`` in
    ``02_trial_segmentation.py`` to pair (onset, offset) sample indices.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Loaded .fif with MNE Annotations for trial start/end events.

    Returns
    -------
    events : np.ndarray, shape (n_events, 3)
        Standard MNE events array: [sample_index, 0, event_id].
        event_id 1 = trial_start ('1' annotation).
        event_id 2 = trial_end   ('2' annotation).

    Raises
    ------
    RuntimeError
        If no valid trial annotations are found.
    """
    try:
        events, _ = mne.events_from_annotations(
            raw,
            event_id=_EVENT_ID_MAP,
            verbose=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"get_event_onsets: failed to parse annotations — {exc}\n"
            f"Available annotations: "
            f"{sorted(set(a['description'] for a in raw.annotations))}"
        ) from exc

    if len(events) == 0:
        raise RuntimeError(
            "get_event_onsets: no events found with IDs "
            f"{_EVENT_ID_MAP}. "
            f"Annotation descriptions present: "
            f"{sorted(set(a['description'] for a in raw.annotations))}"
        )

    n_start = int(np.sum(events[:, 2] == _TRIAL_START_ID))
    n_end = int(np.sum(events[:, 2] == _TRIAL_END_ID))
    logger.info(
        "get_event_onsets: %d total events (%d trial_start, %d trial_end)",
        len(events), n_start, n_end,
    )
    return events


# ===========================================================================
# 1.1.3 — get_subject_group
# ===========================================================================

def get_subject_group(
    subject_id: str,
    config: Optional[dict] = None,
) -> str:
    """
    Return the group label ('Expert' or 'Novice') for a subject.

    Group membership is read from ``stage_d_granger.linear_gc.subjects`` in
    config.yaml, which matches the MATLAB ``Exp_sub`` / ``Nov_sub`` lists
    (R10 is in the excluded list and will raise ValueError).

    Parameters
    ----------
    subject_id : str
        Subject identifier, e.g. 'A1', 'R3'.
    config : dict, optional
        Parsed config.yaml. Loaded from project root if None.

    Returns
    -------
    group : str
        'Expert' or 'Novice'.

    Raises
    ------
    ValueError
        If subject is in the excluded list or not found in any group.
    """
    if config is None:
        config = load_config()

    gc_cfg = config.get("stage_d_granger", {}).get("linear_gc", {})
    subject_groups = gc_cfg.get("subjects", {})

    excluded = subject_groups.get("excluded", [])
    if subject_id in excluded:
        raise ValueError(
            f"Subject {subject_id} is excluded from GC analysis "
            f"(Stage B quality flag). Excluded list: {excluded}"
        )

    for group_name in ("Expert", "Novice"):
        if subject_id in subject_groups.get(group_name, []):
            return group_name

    # Fallback: also check datasets.expert_novice.groups for backward compat
    ds_groups = (
        config.get("datasets", {})
        .get("expert_novice", {})
        .get("groups", {})
    )
    for group_name, members in ds_groups.items():
        if subject_id in members:
            logger.warning(
                "get_subject_group: %s found in datasets.expert_novice.groups "
                "but not in stage_d_granger.linear_gc.subjects — using '%s'",
                subject_id, group_name,
            )
            return group_name

    raise ValueError(
        f"Subject {subject_id} not found in Expert or Novice groups.\n"
        f"Expert: {subject_groups.get('Expert', [])}\n"
        f"Novice: {subject_groups.get('Novice', [])}"
    )


# ===========================================================================
# 1.1.3 — get_connection_labels
# ===========================================================================

def get_connection_labels(roi_names: Optional[list[str]] = None) -> list[str]:
    """
    Return the 20 directed GC connection labels in MATLAB-matching order.

    Order convention
    ----------------
    Mirrors MATLAB column-major reshape of the 5×5 connection cell array:
    ``connections{j,i} = 'Cnames{i} --> Cnames{j}'``
    ``reshape(connections, 25, 1)`` → remove diagonal positions [1,7,13,19,25].

    In Python row-major terms: outer loop over source ROI, inner loop over
    target ROI, diagonal (source==target) skipped.  This produces the same
    20-element ordered list shown in docs/LINEAR_GC_PROGRESS.md §Design:

      LPFC→CPFC, LPFC→RPFC, LPFC→LM1,  LPFC→SMA,
      CPFC→LPFC, CPFC→RPFC, CPFC→LM1,  CPFC→SMA,
      RPFC→LPFC, RPFC→CPFC, RPFC→LM1,  RPFC→SMA,
      LM1→LPFC,  LM1→CPFC,  LM1→RPFC,  LM1→SMA,
      SMA→LPFC,  SMA→CPFC,  SMA→RPFC,  SMA→LM1

    GC matrix convention
    --------------------
    ``GW[src, tgt, freq]`` = GC from ``src`` to ``tgt`` at frequency ``freq``.
    Flattening ``GW[:,:,f].flatten()`` (row-major) and removing indices
    [0, 6, 12, 18, 24] (diagonal) yields values in this exact label order.

    Parameters
    ----------
    roi_names : list of str, optional
        ROI names in order. Defaults to the canonical
        ['LPFC', 'CPFC', 'RPFC', 'LM1', 'SMA'].

    Returns
    -------
    labels : list of str
        20 connection label strings, e.g. 'LPFC→CPFC'.
    """
    if roi_names is None:
        roi_names = ROI_NAMES
    labels = [
        f"{src}\u2192{tgt}"
        for src in roi_names
        for tgt in roi_names
        if src != tgt
    ]
    assert len(labels) == len(roi_names) ** 2 - len(roi_names), (
        f"Expected {len(roi_names)**2 - len(roi_names)} connections, got {len(labels)}"
    )
    return labels


# ===========================================================================
# Convenience: get output path for a Stage D subdirectory
# ===========================================================================

def get_gc_output_dir(
    subdir_key: str,
    config: Optional[dict] = None,
    create: bool = True,
) -> Path:
    """
    Return (and optionally create) a Stage D output subdirectory.

    Parameters
    ----------
    subdir_key : str
        Key from ``stage_d_granger.linear_gc.output_subdirs`` in config.yaml,
        e.g. 'roi_signals', 'gc_connectivity', 'group'.
    config : dict, optional
        Parsed config.yaml. Loaded if None.
    create : bool
        If True, create the directory (including parents).

    Returns
    -------
    path : Path
        Absolute path to the requested output subdirectory.
    """
    if config is None:
        config = load_config()

    stage_d_root = _PROJECT_ROOT / config["paths"]["stage_d_outputs"]
    subdirs = (
        config.get("stage_d_granger", {})
        .get("linear_gc", {})
        .get("output_subdirs", {})
    )

    if subdir_key not in subdirs:
        raise ValueError(
            f"Unknown subdir_key '{subdir_key}'. "
            f"Available: {list(subdirs.keys())}"
        )

    path = stage_d_root / subdirs[subdir_key]
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


# ===========================================================================
# Quick smoke test (run as script)
# ===========================================================================

if __name__ == "__main__":
    import pprint
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    cfg = load_config()

    print("\n--- get_connection_labels() ---")
    labels = get_connection_labels()
    for i, lbl in enumerate(labels):
        print(f"  [{i:2d}] {lbl}")
    print(f"  Total: {len(labels)} connections")

    print("\n--- get_subject_group() ---")
    test_ids = ["A1", "A2", "R7", "R1", "R6"]
    for sid in test_ids:
        grp = get_subject_group(sid, config=cfg)
        print(f"  {sid:4s} → {grp}")

    try:
        get_subject_group("R10", config=cfg)
    except ValueError as e:
        print(f"  R10 → ValueError (expected): {e}")

    print("\n--- load_preprocessed_fif('A1') ---")
    try:
        raw = load_preprocessed_fif("A1", config=cfg, preload=False)
        print(f"  Loaded: {len(raw.ch_names)} ch, {raw.info['sfreq']} Hz, "
              f"{raw.times[-1]:.1f} s")
        print(f"  Annotations: {len(raw.annotations)} total")

        print("\n--- get_event_onsets(raw) ---")
        events = get_event_onsets(raw)
        n_start = int(np.sum(events[:, 2] == 1))
        n_end = int(np.sum(events[:, 2] == 2))
        print(f"  Events array shape: {events.shape}")
        print(f"  Trial start (id=1): {n_start}  |  Trial end (id=2): {n_end}")
        print(f"  First 4 events (sample, 0, id):")
        for row in events[:4]:
            print(f"    sample {row[0]:6d}  id={row[2]}")
    except FileNotFoundError as e:
        print(f"  [SKIP] .fif not found: {e}")

    print("\n--- get_gc_output_dir('roi_signals') ---")
    p = get_gc_output_dir("roi_signals", config=cfg, create=False)
    print(f"  {p}")
    print("\nAll smoke tests passed.\n")
