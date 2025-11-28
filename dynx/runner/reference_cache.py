"""
Reference model caching for metrics with aggressive memory management.

This module provides a caching mechanism to ensure reference models are loaded
only once and shared across all metrics that need them.
"""

import weakref
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from pathlib import Path
import logging
import gc

logger = logging.getLogger(__name__)

class ReferenceModelCache:
    """
    Cache for reference models to avoid repeated loading.
    
    Uses weak references to allow garbage collection when no metrics are using the model.
    """
    
    def __init__(self):
        self._cache: Dict[str, weakref.ref] = {}
        self._strong_refs: Dict[str, Any] = {}  # Keep strong refs during metric runs
        self._access_count: Dict[str, int] = {}  # Track usage
        
    def get_cache_key(self, runner: Any, x: np.ndarray, 
                     requirements: Optional[Dict] = None) -> str:
        """Generate a unique cache key for the reference model."""
        # IMPORTANT: Use the REFERENCE bundle path, not the current method's path!
        # This ensures all fast methods share the same cached reference model
        from .reference_utils import ref_bundle_path
        
        ref_path = ref_bundle_path(runner, x)
        if ref_path:
            key = str(ref_path)
        else:
            # Fallback to ref_params hash if available
            if hasattr(runner, 'ref_params') and runner.ref_params is not None:
                import hashlib
                import pickle
                param_bytes = pickle.dumps(runner.ref_params, protocol=5)
                key = "ref_" + hashlib.md5(param_bytes).hexdigest()[:16]
            else:
                # Last resort: use current params but mark as reference
                import hashlib
                import pickle
                param_bytes = pickle.dumps(x, protocol=5)
                key = "ref_fallback_" + hashlib.md5(param_bytes).hexdigest()[:16]
        
        # For superset caching, we always use the same requirements key
        # This ensures all metrics share the same cached model
        key += "_superset_p0-1"
        
        return key
    
    def _cleanup_model_memory(self, model: Any):
        """Aggressively clean up model memory while keeping essential data."""
        if model is None:
            return
            
        try:
            # Clean up large arrays we don't need for metrics
            for period in getattr(model, 'periods_list', []):
                for stage in getattr(period, 'stages', {}).values():
                    for perch_name, perch in getattr(stage, 'perches', {}).items():
                        if hasattr(perch, 'sol') and perch.sol is not None:
                            # For reference models, we typically only need policy arrays
                            # Clear Q, lambda, and other large arrays
                            if hasattr(perch.sol, '_jit'):
                                # Keep policy but clear value/Q functions
                                if hasattr(perch.sol._jit, 'Q'):
                                    perch.sol._jit.Q = None
                                if hasattr(perch.sol._jit, 'lambda_'):
                                    perch.sol._jit.lambda_ = None
                                if hasattr(perch.sol._jit, 'vlu'):
                                    perch.sol._jit.vlu = None
                            
                            # Clear EGM data if not needed
                            if hasattr(perch.sol, 'EGM'):
                                perch.sol.EGM = None
                                
            gc.collect()
            
        except Exception as e:
            logger.warning(f"Error during model cleanup: {e}")
    
    def get(self, runner: Any, x: np.ndarray, 
            metric_requirements: Optional[Dict] = None,
            cleanup_for_metrics: bool = True) -> Optional[Any]:
        """
        Get reference model from cache or load it.
        
        Args:
            runner: CircuitRunner instance
            x: Parameter vector
            metric_requirements: Optional requirements for selective loading (ignored for superset)
            cleanup_for_metrics: Clean up non-essential data for metrics
            
        Returns:
            Reference model or None
        """
        # For superset caching, always load the maximum set needed by any metric
        superset_requirements = self._get_superset_requirements()
        cache_key = self.get_cache_key(runner, x, superset_requirements)
        
        # Check if we have a cached model
        if cache_key in self._cache:
            ref_weak = self._cache[cache_key]
            model = ref_weak()
            if model is not None:
                logger.info(f"Using cached reference model for {cache_key}")
                self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
                # Log which fast method is using the cached reference
                if hasattr(runner, 'unpack'):
                    params = runner.unpack(x)
                    method = params.get('master.methods.upper_envelope', 'unknown')
                    logger.info(f"  Fast method '{method}' using cached reference model")
                return model
            else:
                # Weak reference died, remove from cache
                del self._cache[cache_key]
                if cache_key in self._access_count:
                    del self._access_count[cache_key]
        
        # Load the model with superset requirements
        logger.info(f"Loading reference model for {cache_key} with superset requirements")
        from dynx.runner.reference_utils import load_reference_model as _load_reference_model
        
        model = _load_reference_model(runner, x, superset_requirements)
        if model is not None:
            # Clean up memory if requested
            if cleanup_for_metrics:
                logger.info("Cleaning up reference model memory for metrics")
                self._cleanup_model_memory(model)
            
            # Store weak reference
            self._cache[cache_key] = weakref.ref(model)
            # Also keep a strong reference during metric computation
            self._strong_refs[cache_key] = model
            self._access_count[cache_key] = 1
            
        return model
    
    def clear_strong_refs(self):
        """Clear strong references to allow garbage collection."""
        logger.info(f"Clearing {len(self._strong_refs)} strong references")
        for key, count in self._access_count.items():
            logger.debug(f"  {key}: accessed {count} times")
        self._strong_refs.clear()
        gc.collect()
        
    def clear(self):
        """Clear the entire cache."""
        self._cache.clear()
        self._strong_refs.clear()
        self._access_count.clear()
        gc.collect()

# Global cache instance
_reference_cache = ReferenceModelCache()

def get_cached_reference_model(runner: Any, x: np.ndarray, 
                             metric_requirements: Optional[Dict] = None) -> Optional[Any]:
    """
    Get reference model using cache with aggressive memory cleanup.
    
    This is a drop-in replacement for load_reference_model that adds caching
    and memory optimization for metrics.
    
    NOTE: This now delegates to the unified model cache which can share
    models between baseline runs and reference lookups.
    """
    # Try unified cache first
    try:
        from .model_cache import get_cached_reference_model as _get_unified
        return _get_unified(runner, x, metric_requirements)
    except ImportError:
        # Fall back to local cache if unified cache not available
        return _reference_cache.get(runner, x, metric_requirements, cleanup_for_metrics=True)

def clear_reference_cache():
    """Clear the reference model cache."""
    _reference_cache.clear()

def release_strong_references():
    """
    Release strong references to allow garbage collection.
    
    Call this after all metrics have been computed.
    """
    _reference_cache.clear_strong_refs()
    
    def _get_superset_requirements(self) -> Dict:
        """
        Get the superset of all metric requirements.
        
        This returns the union of all periods and stages needed by any metric,
        ensuring a single cached model can serve all metrics.
        """
        # Based on analysis of METRIC_REQUIREMENTS:
        # - Euler error needs periods [0, 1]
        # - All other metrics only need period 0
        # - Period 0 needs union of stages: OWNC, OWNH, TENU, RNTH, RNTC
        # - Period 1 needs all stages (None)
        return {
            'periods_to_load': [0, 1],
            'stages_to_load': {
                0: ['OWNC', 'OWNH', 'TENU', 'RNTH', 'RNTC'],  # Union of all period 0 stages
                1: None  # All stages for period 1 (required by euler_error)
            }
        }

def get_cache_stats():
    """Get statistics about cache usage."""
    return {
        'cached_models': len(_reference_cache._cache),
        'strong_refs': len(_reference_cache._strong_refs),
        'access_counts': dict(_reference_cache._access_count)
    }