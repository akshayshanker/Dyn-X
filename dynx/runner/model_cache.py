"""
Unified model caching for both baseline models and reference models.

This module provides a unified cache that can be used for:
1. Baseline models loaded as part of the method list (when --include-baseline)
2. Reference models loaded for comparison metrics
3. Any other models that need to be shared across components

The key insight is that the baseline model (e.g., VFI_HD_GRID) loaded with
periods 0 and 1 is exactly the same as the reference model needed for metrics.
"""

import weakref
from typing import Any, Dict, Optional, Tuple, List
import numpy as np
from pathlib import Path
import logging
import gc

logger = logging.getLogger(__name__)

class UnifiedModelCache:
    """
    Unified cache for models to avoid repeated loading.
    
    This cache can store:
    - Baseline models loaded as methods
    - Reference models for metrics
    - Any other models that should be shared
    """
    
    def __init__(self):
        self._cache: Dict[str, weakref.ref] = {}
        self._strong_refs: Dict[str, Any] = {}
        self._access_count: Dict[str, int] = {}
        self._method_to_model: Dict[str, str] = {}  # Map method names to cache keys
        
    def get_model_cache_key(self, model_id: str, periods: Optional[List[int]] = None) -> str:
        """
        Generate a cache key for a model.
        
        Args:
            model_id: Model identifier (e.g., bundle path or method name)
            periods: Optional list of periods for selective loading
            
        Returns:
            Cache key string
        """
        if periods is None:
            # Full model with all periods
            return f"{model_id}_full"
        else:
            # Selective load with specific periods
            periods_str = "-".join(map(str, sorted(periods)))
            return f"{model_id}_p{periods_str}"
    
    def register_method_model(self, method: str, model: Any, periods: Optional[List[int]] = None):
        """
        Register a model that was loaded for a specific method.
        
        This is called when a method (e.g., VFI_HD_GRID) loads its model,
        allowing it to be reused as a reference model.
        
        Args:
            method: Method name (e.g., "VFI_HD_GRID")
            model: The loaded model
            periods: Periods that were loaded
        """
        # Generate cache key based on the model's bundle path if available
        # Look for bundle path in various places
        if hasattr(model, '_bundle_path'):
            model_id = str(model._bundle_path)
        elif hasattr(model, 'bundle_path'):
            model_id = str(model.bundle_path)
        else:
            # Try to get from the method's expected path
            model_id = f"{method}_bundle"
            
        cache_key = self.get_model_cache_key(model_id, periods)
        
        # Store in cache
        self._cache[cache_key] = weakref.ref(model)
        self._strong_refs[cache_key] = model
        self._access_count[cache_key] = 1
        self._method_to_model[method] = cache_key
        
        logger.info(f"Registered model for method '{method}' with cache key '{cache_key}'")
        
        # Also register the superset cache key if this is the baseline with periods 0,1
        if periods == [0, 1] or periods is None:
            superset_key = self.get_model_cache_key(model_id, [0, 1]) + "_superset_p0-1"
            self._cache[superset_key] = weakref.ref(model)
            logger.info(f"Also registered as superset reference with key '{superset_key}'")
    
    def get_reference_model(self, runner: Any, x: np.ndarray, 
                          requirements: Optional[Dict] = None) -> Optional[Any]:
        """
        Get reference model, checking if baseline was already loaded.
        
        This first checks if the baseline model was already loaded as a method,
        and if so, returns it. Otherwise falls back to loading it.
        
        Args:
            runner: CircuitRunner instance
            x: Parameter vector
            requirements: Optional loading requirements
            
        Returns:
            Reference model or None
        """
        # First check if we have the baseline cached from method execution
        baseline_method = getattr(runner, 'baseline_method', 'VFI_HD_GRID')
        if baseline_method in self._method_to_model:
            cache_key = self._method_to_model[baseline_method]
            if cache_key in self._cache:
                ref_weak = self._cache[cache_key]
                model = ref_weak()
                if model is not None:
                    logger.info(f"Using baseline model '{baseline_method}' as reference (from method cache)")
                    self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
                    return model
        
        # Otherwise, check if we have it cached with the reference path
        from .reference_utils import ref_bundle_path
        ref_path = ref_bundle_path(runner, x)
        if ref_path:
            # Try superset cache key
            cache_key = self.get_model_cache_key(str(ref_path), [0, 1]) + "_superset_p0-1"
            if cache_key in self._cache:
                ref_weak = self._cache[cache_key]
                model = ref_weak()
                if model is not None:
                    logger.info(f"Using cached reference model with key '{cache_key}'")
                    self._access_count[cache_key] = self._access_count.get(cache_key, 0) + 1
                    return model
        
        # If not cached, load it
        logger.info("Reference model not in cache, loading...")
        from .reference_utils import load_reference_model
        
        # Always load superset for reference
        superset_requirements = {
            'periods_to_load': [0, 1],
            'stages_to_load': {
                0: ['OWNC', 'OWNH', 'TENU', 'RNTH', 'RNTC'],
                1: None
            }
        }
        
        model = load_reference_model(runner, x, superset_requirements)
        if model is not None and ref_path:
            # Cache it
            cache_key = self.get_model_cache_key(str(ref_path), [0, 1]) + "_superset_p0-1"
            self._cache[cache_key] = weakref.ref(model)
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
        self._method_to_model.clear()
        gc.collect()
        
    def get_stats(self):
        """Get cache statistics."""
        return {
            'cached_models': len(self._cache),
            'strong_refs': len(self._strong_refs),
            'method_models': len(self._method_to_model),
            'access_counts': dict(self._access_count)
        }

# Global unified cache instance
_unified_cache = UnifiedModelCache()

def register_baseline_model(method: str, model: Any, periods: Optional[List[int]] = None):
    """
    Register a baseline model that was loaded for a method.
    
    This should be called when a baseline method (e.g., VFI_HD_GRID) loads
    its model, allowing it to be reused as the reference model for metrics.
    """
    _unified_cache.register_method_model(method, model, periods)

def get_cached_reference_model(runner: Any, x: np.ndarray, 
                             metric_requirements: Optional[Dict] = None) -> Optional[Any]:
    """
    Get reference model using unified cache.
    
    This checks if the baseline was already loaded and reuses it,
    otherwise loads and caches it.
    """
    return _unified_cache.get_reference_model(runner, x, metric_requirements)

def clear_model_cache():
    """Clear the unified model cache."""
    _unified_cache.clear()

def release_model_references():
    """Release strong references to allow garbage collection."""
    _unified_cache.clear_strong_refs()

def get_cache_stats():
    """Get statistics about cache usage."""
    return _unified_cache.get_stats()