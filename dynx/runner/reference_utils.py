"""
Reference model utilities for loading previously solved models.

This module provides functions to locate and load reference models
that have been previously solved and saved as bundles.
"""

from pathlib import Path
from typing import Any, Optional, Dict, List
import warnings
import numpy as np

from dynx.runner.circuit_runner import CircuitRunner

# Default method to use as reference
DEFAULT_REF_METHOD = "VFI_HDGRID"


def ref_bundle_path(runner: CircuitRunner, x: np.ndarray) -> Optional[Path]:
    """
    Get path to reference bundle directory.

    If runner.ref_params exists, it uses those parameters to construct the
    path. Otherwise, it falls back to substituting only the method name,
    which may fail if other parameters like grid sizes differ.

    Args:
        runner: CircuitRunner instance
        x: Parameter vector of the current (non-reference) run

    Returns:
        Path to reference bundle directory, or None if no output_root
    """
    if not runner.output_root:
        return None

    if hasattr(runner, 'ref_params') and runner.ref_params is not None:
        # If reference parameters are stored on the runner, use them directly.
        # This is the most reliable way to find the baseline bundle.
        x_ref = runner.ref_params
    else:
        # Fallback for older runners or cases where ref_params isn't set:
        # substitute method name only. This is brittle.
        warnings.warn("runner.ref_params not found. Falling back to method substitution. "
                      "This may fail to find the baseline if grid parameters differ.")
        if runner.method_param_path and runner.method_param_path in runner.param_paths:
            x_ref = x.copy()
            method_idx = runner.param_paths.index(runner.method_param_path)
            x_ref[method_idx] = DEFAULT_REF_METHOD
        else:
            x_ref = x

    return runner._bundle_path(x_ref)


def load_reference_model(
    runner: CircuitRunner, 
    x: np.ndarray,
    metric_requirements: Optional[dict] = None
) -> Optional[Any]:
    """
    Load reference model if it exists, with optional selective loading.

    Looks for a bundle saved with DEFAULT_REF_METHOD for the given parameters.

    Args:
        runner: CircuitRunner instance
        x: Parameter vector
        metric_requirements: Optional dict with keys:
            - periods_to_load: List[int] of periods needed
            - stages_to_load: Dict[int, List[str]] of stages per period

    Returns:
        ModelCircuit if reference exists and loads successfully, None otherwise
    """
    ref_path = ref_bundle_path(runner, x)
    ref_cfg = runner.patch_config(runner.ref_params)
    
    if not ref_path:
        return None
    
    # If we have metric requirements, use selective loading
    if metric_requirements:
        periods_to_load = metric_requirements.get('periods_to_load')
        stages_to_load = metric_requirements.get('stages_to_load')
        return runner._maybe_load_bundle(
            ref_path, 
            ref_cfg,
            periods_to_load=periods_to_load,
            stages_to_load=stages_to_load
        )
    
    # Otherwise fall back to full load
    return runner._maybe_load_bundle(ref_path, ref_cfg)