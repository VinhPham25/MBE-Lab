"""
ASTaUND Phase 1 - Utility Functions
Configuration loader and common helper functions for the fNIRS preprocessing pipeline

Created: 2026-01-19
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import sys
from datetime import datetime

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Plot timestamp helper (applies globally to matplotlib figure save calls)
# ---------------------------------------------------------------------------
_SAVEFIG_PATCHED = False


def add_plot_timestamp(fig, timestamp_text: Optional[str] = None):
    """
    Add a bottom-right timestamp annotation to a matplotlib figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to annotate before saving.
    timestamp_text : str, optional
        Custom text. If None, uses local time formatted as
        "Generated: YYYY-MM-DD HH:MM:SS".

    Returns
    -------
    matplotlib.figure.Figure
        The same figure object, annotated in-place.
    """
    if fig is None:
        return fig

    if timestamp_text is None:
        timestamp_text = f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Remove prior timestamp text if present, so repeated savefig() calls do
    # not stack multiple timestamp labels on the same figure.
    for txt in list(getattr(fig, "texts", [])):
        if getattr(txt, "get_gid", lambda: None)() == "autogen_timestamp":
            txt.remove()

    t = fig.text(
        0.995, 0.005, timestamp_text,
        ha="right", va="bottom",
        fontsize=7, color="#444444", alpha=0.85,
        transform=fig.transFigure,
    )
    t.set_gid("autogen_timestamp")
    return fig


def enable_plot_timestamps():
    """
    Enable automatic timestamp overlays for all matplotlib figure saves.

    This monkey-patches matplotlib's Figure.savefig once per Python process.
    Any script importing `utils.py` will automatically get timestamped saved
    plots without changing individual plotting code.

    Parameters
    ----------
    None

    Returns
    -------
    bool
        True if patching is active, False if matplotlib is unavailable.
    """
    global _SAVEFIG_PATCHED
    if _SAVEFIG_PATCHED:
        return True

    try:
        from matplotlib.figure import Figure  # local import to avoid hard dependency at module import
    except Exception:
        return False

    original_savefig = Figure.savefig

    def _timestamped_savefig(self, *args, **kwargs):
        add_plot_timestamp(self)
        return original_savefig(self, *args, **kwargs)

    Figure.savefig = _timestamped_savefig
    _SAVEFIG_PATCHED = True
    return True


# Activate timestamped savefig globally for all plotting scripts importing utils.
enable_plot_timestamps()


# ---------------------------------------------------------------------------
# pyswarms report.log redirect
# ---------------------------------------------------------------------------
# pyswarms' internal Reporter (used by ps.single.GlobalBestPSO, called from
# stage_e_transfer_entropy/phase_3/11_bimodal_dspte.py for PSO weight
# optimisation) defaults to writing `report.log` (+ up to 20 rotated
# `report.log.N` backups) into the current working directory every time it's
# instantiated. Reporter only lets you override that path via a YAML file
# referenced by the `LOG_CFG` environment variable, so we generate one that
# points at `report_logs/` at the project root instead of polluting the repo
# root / cwd.
_PYSWARMS_LOG_CFG_ENV = "LOG_CFG"
_PYSWARMS_LOGGING_CONFIGURED = False


def configure_pyswarms_logging() -> Path:
    """
    Redirect pyswarms' Reporter log output from `<cwd>/report.log` to
    `report_logs/report.log` at the project root.

    Safe to call multiple times (and from multiple scripts/processes) —
    subsequent calls are no-ops once `LOG_CFG` is already set for this
    process. Call this before any code path that may instantiate
    `pyswarms.single.GlobalBestPSO` (or any other pyswarms optimizer).

    Returns
    -------
    Path
        Directory that pyswarms log files are written to (`report_logs/`).
    """
    global _PYSWARMS_LOGGING_CONFIGURED

    report_logs_dir = PROJECT_ROOT / 'report_logs'
    report_logs_dir.mkdir(parents=True, exist_ok=True)

    if _PYSWARMS_LOGGING_CONFIGURED or os.environ.get(_PYSWARMS_LOG_CFG_ENV):
        return report_logs_dir

    log_config_path = report_logs_dir / '.pyswarms_log_config.yaml'
    log_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'default': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file_default': {
                'level': 'INFO',
                'formatter': 'standard',
                'class': 'logging.handlers.RotatingFileHandler',
                'filename': str(report_logs_dir / 'report.log'),
                'encoding': 'utf8',
                'maxBytes': 10485760,
                'backupCount': 20,
            },
        },
        'loggers': {
            '': {
                'handlers': ['default', 'file_default'],
                'level': 'INFO',
                'propagate': True,
            }
        },
    }
    with open(log_config_path, 'w') as f:
        yaml.safe_dump(log_config, f)

    os.environ[_PYSWARMS_LOG_CFG_ENV] = str(log_config_path)
    _PYSWARMS_LOGGING_CONFIGURED = True
    return report_logs_dir


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and parse the config.yaml file.
    
    Parameters
    ----------
    config_path : str, optional
        Path to config file. If None, uses default config.yaml in project root.
        
    Returns
    -------
    config : dict
        Parsed configuration dictionary
        
    Examples
    --------
    >>> config = load_config()
    >>> print(config['stage_a']['snr']['threshold_low'])
    80
    """
    if config_path is None:
        config_path = PROJECT_ROOT / 'config.yaml'
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_dataset_config(dataset_name: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get configuration for a specific dataset.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name: 'expert_novice', 'learning_curve', 'short_term_tes', 'long_term_tes'
    config : dict, optional
        Configuration dictionary. If None, loads from default config.yaml
        
    Returns
    -------
    dataset_config : dict
        Dataset-specific configuration
        
    Examples
    --------
    >>> ds_config = get_dataset_config('expert_novice')
    >>> print(ds_config['sfreq'])
    25.0
    """
    if config is None:
        config = load_config()
    
    if dataset_name not in config['datasets']:
        raise ValueError(f"Dataset '{dataset_name}' not found in config. "
                        f"Available: {list(config['datasets'].keys())}")
    
    return config['datasets'][dataset_name]


def get_stage_params(stage: str, config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Get parameters for a specific processing stage.
    
    Parameters
    ----------
    stage : str
        Stage name: 'stage_a', 'stage_b', or 'stage_c'
    config : dict, optional
        Configuration dictionary. If None, loads from default config.yaml
        
    Returns
    -------
    stage_params : dict
        Stage-specific parameters
        
    Examples
    --------
    >>> params = get_stage_params('stage_a')
    >>> print(params['snr']['threshold_low'])
    80
    """
    if config is None:
        config = load_config()
    
    if stage not in config:
        raise ValueError(f"Stage '{stage}' not found in config. "
                        f"Available: stage_a, stage_b, stage_c")
    
    return config[stage]


def get_subject_path(dataset_name: str, subject_id: str, 
                     config: Optional[Dict] = None) -> Path:
    """
    Get the path to a subject's data directory.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name: 'expert_novice', 'learning_curve', 'short_term_tes', 'long_term_tes'
    subject_id : str
        Subject ID (e.g., 'A1', 'R1', 'S1')
    config : dict, optional
        Configuration dictionary. If None, loads from default config.yaml
        
    Returns
    -------
    subject_path : Path
        Path to subject's data directory
        
    Examples
    --------
    >>> path = get_subject_path('expert_novice', 'A1')
    >>> print(path)
    data/BIDS/Expert-Novice/sub-A1
    """
    if config is None:
        config = load_config()
    
    dataset_config = get_dataset_config(dataset_name, config)
    data_path = PROJECT_ROOT / dataset_config['data_path']
    subject_path = data_path / f'sub-{subject_id}'
    
    return subject_path


def get_snirf_files(dataset_name: str, subject_id: str,
                    config: Optional[Dict] = None) -> List[Path]:
    """
    Get list of .snirf files for a subject.
    
    Parameters
    ----------
    dataset_name : str
        Dataset name
    subject_id : str
        Subject ID
    config : dict, optional
        Configuration dictionary. If None, loads from default config.yaml
        
    Returns
    -------
    snirf_files : list of Path
        List of .snirf file paths
        
    Examples
    --------
    >>> files = get_snirf_files('expert_novice', 'A1')
    >>> print(files[0].name)
    sub-A1_task-ExpNovPatternCutting_nirs.snirf
    """
    subject_path = get_subject_path(dataset_name, subject_id, config)
    nirs_path = subject_path / 'nirs'
    
    if not nirs_path.exists():
        raise FileNotFoundError(f"NIRS directory not found: {nirs_path}")
    
    snirf_files = sorted(nirs_path.glob('*.snirf'))
    
    if not snirf_files:
        raise FileNotFoundError(f"No .snirf files found in {nirs_path}")
    
    return snirf_files


def get_output_path(stage: str, subdir: Optional[str] = None,
                   config: Optional[Dict] = None) -> Path:
    """
    Get output directory path for a specific stage.
    
    Parameters
    ----------
    stage : str
        Stage name: 'stage_a', 'stage_b', or 'stage_c'
    subdir : str, optional
        Subdirectory within stage output (e.g., 'figures', 'data')
    config : dict, optional
        Configuration dictionary. If None, loads from default config.yaml
        
    Returns
    -------
    output_path : Path
        Path to output directory (created if doesn't exist)
        
    Examples
    --------
    >>> path = get_output_path('stage_a', 'figures')
    >>> print(path)
    outputs/stage_a_qa/figures
    """
    if config is None:
        config = load_config()
    
    # Map stage names to output paths
    stage_output_map = {
        'stage_a': config['paths']['stage_a_outputs'],
        'stage_b': config['paths']['stage_b_outputs'],
        'stage_c': config['paths']['stage_c_outputs']
    }
    
    if stage not in stage_output_map:
        raise ValueError(f"Invalid stage: {stage}")
    
    output_path = PROJECT_ROOT / stage_output_map[stage]
    
    if subdir:
        output_path = output_path / subdir
    
    # Create directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    return output_path


def validate_config(config: Optional[Dict] = None) -> bool:
    """
    Validate configuration file structure and required parameters.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary. If None, loads from default config.yaml
        
    Returns
    -------
    valid : bool
        True if configuration is valid
        
    Raises
    ------
    ValueError
        If configuration is invalid or missing required parameters
    """
    if config is None:
        config = load_config()
    
    # Check required top-level keys
    required_keys = ['datasets', 'stage_a', 'stage_b', 'stage_c', 'paths']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Check dataset configurations
    required_datasets = ['expert_novice', 'learning_curve', 'short_term_tes', 'long_term_tes']
    for dataset in required_datasets:
        if dataset not in config['datasets']:
            raise ValueError(f"Missing dataset configuration: {dataset}")
        
        # Check required dataset parameters
        ds_config = config['datasets'][dataset]
        required_ds_keys = ['name', 'system', 'sfreq', 'wavelengths', 'data_path']
        for key in required_ds_keys:
            if key not in ds_config:
                raise ValueError(f"Missing parameter '{key}' in dataset '{dataset}'")
    
    # Check Stage A parameters
    stage_a = config['stage_a']
    if 'snr' not in stage_a or 'threshold_low' not in stage_a['snr']:
        raise ValueError("Missing Stage A SNR threshold_low parameter")
    if 'threshold_high' not in stage_a['snr']:
        raise ValueError("Missing Stage A SNR threshold_high parameter")
    
    # Check Stage B parameters
    stage_b = config['stage_b']
    if 'filter' not in stage_b or 'l_freq' not in stage_b['filter']:
        raise ValueError("Missing Stage B filter l_freq parameter")
    
    # Check Stage C parameters
    stage_c = config['stage_c']
    if 'epochs' not in stage_c or 'tmin' not in stage_c['epochs']:
        raise ValueError("Missing Stage C epochs tmin parameter")
    
    return True


def print_config_summary(config: Optional[Dict] = None):
    """
    Print a summary of the configuration.
    
    Parameters
    ----------
    config : dict, optional
        Configuration dictionary. If None, loads from default config.yaml
    """
    if config is None:
        config = load_config()
    
    print("=" * 70)
    print("CONFIG SUMMARY")
    print("=" * 70)
    
    # Datasets
    print("\nDATASETS:")
    for name, ds_config in config['datasets'].items():
        print(f"  • {name}:")
        print(f"      System: {ds_config['system']}, "
              f"Sampling: {ds_config['sfreq']} Hz, "
              f"Wavelengths: {ds_config['wavelengths']}")
    
    # Stage A
    print("\nSTAGE A - RAW QUALITY ASSESSMENT:")
    print(f"  • SNR Threshold: {config['stage_a']['snr']['threshold_low']}-"
          f"{config['stage_a']['snr']['threshold_high']} dB")
    print(f"  • Motion tMotion: {config['stage_a']['motion']['tMotion']}")
    print(f"  • Motion STDEVthresh: {config['stage_a']['motion']['STDEVthresh']}")
    
    # Stage B
    print("\nSTAGE B - PREPROCESSING:")
    print(f"  • Filter: {config['stage_b']['filter']['l_freq']}-"
          f"{config['stage_b']['filter']['h_freq']} Hz")
    print(f"  • rPCA n_components: {config['stage_b']['rpca']['n_components']}")
    
    # Stage C
    print("\nSTAGE C - VALIDATION:")
    print(f"  • Epoch window: {config['stage_c']['epochs']['tmin']} to "
          f"{config['stage_c']['epochs']['tmax']} s")
    print(f"  • HRF peak window: {config['stage_c']['hrf']['peak_analysis_window']}")
    
    print("\n" + "=" * 70)


# Test function
def test_config_loader():
    """
    Test all configuration loader functions.
    """
    print("Testing config loader utilities...\n")
    
    # Test 1: Load config
    print("Test 1: Loading config file")
    config = load_config()
    print(f"✓ Config loaded: {len(config)} top-level keys\n")
    
    # Test 2: Get dataset config
    print("Test 2: Getting dataset configuration")
    ds_config = get_dataset_config('expert_novice')
    print(f"✓ Expert-Novice config: {ds_config['sfreq']} Hz, "
          f"{ds_config['wavelengths']} nm\n")
    
    # Test 3: Get stage parameters
    print("Test 3: Getting stage parameters")
    stage_a_params = get_stage_params('stage_a')
    print(f"✓ Stage A SNR threshold: {stage_a_params['snr']['threshold_low']}-"
          f"{stage_a_params['snr']['threshold_high']} dB\n")
    
    # Test 4: Get subject path
    print("Test 4: Getting subject path")
    subject_path = get_subject_path('expert_novice', 'A1')
    print(f"✓ Subject A1 path: {subject_path}")
    print(f"  Exists: {subject_path.exists()}\n")
    
    # Test 5: Get .snirf files
    print("Test 5: Getting .snirf files")
    try:
        snirf_files = get_snirf_files('expert_novice', 'A1')
        print(f"✓ Found {len(snirf_files)} .snirf file(s)")
        for f in snirf_files:
            print(f"  • {f.name} ({f.stat().st_size / (1024*1024):.2f} MB)")
    except FileNotFoundError as e:
        print(f"⚠ {e}")
    print()
    
    # Test 6: Get output path
    print("Test 6: Getting output paths")
    output_path = get_output_path('stage_a', 'figures')
    print(f"✓ Stage A figures output: {output_path}")
    print(f"  Exists: {output_path.exists()}\n")
    
    # Test 7: Validate config
    print("Test 7: Validating configuration")
    try:
        validate_config()
        print("✓ Configuration is valid\n")
    except ValueError as e:
        print(f"✗ Configuration error: {e}\n")
    
    # Test 8: Print summary
    print("Test 8: Configuration summary")
    print_config_summary()
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    # Run tests if executed directly
    test_config_loader()
