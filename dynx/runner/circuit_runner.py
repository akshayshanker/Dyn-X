"""
Circuit Runner for parameter sweeps of economic models.

This module provides the CircuitRunner class, which prepares configurations
for parameter sweeps. It supports patching configurations with parameter
values, building and running models, and collecting metrics. It also includes
utilities for parallel execution with MPI.

Note — samplers should use `np.random.default_rng` for side-effect-free randomness.
"""

import json
import pickle
import copy
import hashlib
import warnings
import inspect
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml

try:
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False

from dynx.runner.telemetry import RunRecorder

# Import save/load functions and compile_all_stages
try:
    from dynx.stagecraft.io import save_circuit, load_circuit
    from dynx.stagecraft.makemod import compile_all_stages

    HAS_IO = True
except ImportError:
    HAS_IO = False
    warnings.warn("dynx.stagecraft.io not available; save/load functionality disabled")


def set_deep(d: Dict[str, Any], path: str, val: Any) -> None:
    """
    Set a value in a nested dictionary, creating intermediate dictionaries if needed.

    Args:
        d: Dictionary to modify
        path: Path in the form "key1.key2.key3" specifying the location in the dictionary
        val: Value to set at the path

    Examples:
        >>> d = {}
        >>> set_deep(d, "a.b.c", 42)
        >>> d
        {'a': {'b': {'c': 42}}}
    """
    keys = path.split(".")
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = val


class CircuitRunner:
    """
    Run economic models with parameter sweeps.

    CircuitRunner is designed to efficiently run economic models with different
    parameter values. It handles configuration patching and metric collection.

    Supports disk-based save/load functionality to avoid re-solving models
    with identical parameters.

    Attributes:
        base_cfg: Base configuration dictionary for the model
        param_paths: List of parameter paths to patch (in the form "path.to.param")
        model_factory: Function to create model instances from a configuration
        solver: Function to solve a model
        simulator: Optional function to simulate a solved model
        metric_fns: Optional dictionary of functions that compute metrics from a model
        cache: Whether to cache results based on parameter values
        output_root: Parent directory where bundles are written/searched
        bundle_prefix: First part of every bundle folder name
        save_by_default: If True, every solved model is persisted
        load_if_exists: If True, run() will look for existing bundles and load instead of solving
        hash_len: Number of hex chars to keep from MD5 hash
        _cache: Internal cache of (parameter vector, metrics) pairs
        method_param_path: Path to parameter that determines method subdirectory for bundles.
                          Default is "master.methods.upper_envelope". Set to None to disable
                          method-based bundle organization (stores bundles flat).
    """

    def __init__(
        self,
        base_cfg: Dict[str, Any],
        param_paths: List[str],
        model_factory: Callable[[Dict[str, Any]], Any],
        solver: Callable[[Any], None],
        metric_fns: Optional[Dict[str, Callable[[Any], float]]] = None,
        simulator: Optional[Callable[[Any], None]] = None,
        sampler: Optional[Any] = None,
        cache: bool = False,
        validate_paths: bool = False,
        # New bundle management parameters
        output_root: Optional[Union[str, Path]] = None,
        bundle_prefix: str = "run",
        save_by_default: bool = False,
        load_if_exists: bool = False,
        hash_len: int = 8,
        config_src: Optional[Union[str, Path, Dict[str, Any], List[Any]]] = None,
        method_param_path: Optional[str] = "master.methods.upper_envelope",
    ):
        """
        Initialize a CircuitRunner.

        Args:
            base_cfg: Base configuration dictionary for the model
            param_paths: List of parameter paths to patch (in the form "path.to.param")
            model_factory: Function that creates model instances from a configuration
            solver: Function that solves a model
            metric_fns: Optional dictionary of functions that compute metrics from a model
            simulator: Optional function that simulates a solved model
            sampler: Optional sampler to use for generating parameter values (deprecated)
            cache: Whether to cache results based on parameter values
            validate_paths: Whether to validate that parameter paths exist in the configuration
            output_root: Parent directory for bundle storage. If None, saving is disabled unless save_model=True is passed to run()
            bundle_prefix: First part of every bundle folder name
            save_by_default: If True, every solved model is persisted unless explicitly overridden at call-time
            load_if_exists: If True, run() will first look for an existing bundle and load it instead of solving
            hash_len: Number of hex chars to keep from the MD5 hash of the parameter vector
            config_src: Source configuration to pass to save_circuit. If None, uses base_cfg
            method_param_path: Path to parameter that determines method subdirectory for bundles.
                              Default is "master.methods.upper_envelope". Set to None to disable
                              method-based bundle organization (stores bundles flat).
        """
        if sampler is not None:
            warnings.warn(
                "`sampler` argument is deprecated and will be removed in v1.7; "
                "draw xs with dynx.runner.sampler.build_design() instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        self.base_cfg = copy.deepcopy(base_cfg)
        self.param_paths = param_paths
        self.model_factory = model_factory
        self.solver = solver
        self.metric_fns = metric_fns or {}
        self.simulator = simulator
        self.cache = cache
        self._cache = {} if cache else None

        # Bundle management attributes
        self.output_root = Path(output_root).expanduser().resolve() if output_root else None
        self.bundle_prefix = bundle_prefix
        self.save_by_default = save_by_default
        self.load_if_exists = load_if_exists
        self.hash_len = hash_len
        self.config_src = config_src  # Store original config source for saving
        self.method_param_path = method_param_path

        # Validate parameter paths if requested
        if validate_paths:
            self._validate_param_paths()

        # Check if IO functionality is available when bundle features are requested
        if (self.output_root or save_by_default or load_if_exists) and not HAS_IO:
            warnings.warn(
                "Bundle save/load functionality requires dynx.stagecraft.io but it's not available. "
                "Bundle features will be disabled."
            )

    def _validate_param_paths(self) -> None:
        """
        Validate that all parameter paths exist in the configuration.

        Raises:
            ValueError: If a parameter path does not exist in the base configuration.
        """
        for path in self.param_paths:
            keys = path.split(".")
            d = self.base_cfg

            try:
                for key in keys[:-1]:
                    d = d[key]

                # Check if final key exists (but don't access it yet)
                if keys[-1] not in d:
                    raise KeyError

            except (KeyError, TypeError):
                raise ValueError(f"Parameter path '{path}' not found in configuration")

    # ------------------------------------------------------------------
    # ░░  Hashing helper  ░░
    # ------------------------------------------------------------------
    def _hash_param_vec(self, x: np.ndarray) -> str:
        """
        Compute the bundle-hash for a parameter vector *x*.

        • Always ignore the auxiliary “__runner.mode” flag  
          (it only controls load/solve behaviour).  
        • If *method_param_path* is set, also ignore that entry so that
          VFI_HDGRID, FUES, … share one hash when the *other* parameters
          are identical.

        Everything else stays in the hash, so changing, say,
        a preference parameter still yields a different directory.

        Returns
        -------
        str   First ``self.hash_len`` hexadecimal digits of MD5 digest.
        """
        # 1. Build a dictionary for ease of access
        param_dict = self.unpack(x)

        # 2. Build the list of parameter-paths we want to keep
        ignore = {"__runner.mode"}
        if self.method_param_path:
            ignore.add(self.method_param_path)

        keep_paths = [p for p in self.param_paths if p not in ignore]

        # 3. Assemble the *filtered* vector used for hashing
        if keep_paths:
            vals_for_hash = np.array([param_dict[p] for p in keep_paths], dtype=object)
        else:
            # Edge-case: we ignored every path ⇒ hash an empty byte-string
            vals_for_hash = np.array([], dtype=object)

        # 4. Feed into MD5
        key_bytes = pickle.dumps(vals_for_hash, protocol=5)
        digest = hashlib.md5(key_bytes).hexdigest()

        return digest[: self.hash_len]


    def _bundle_path(self, x: np.ndarray) -> Optional[Path]:
        """
        Build directory path for bundle using output_root, bundle_prefix, and hash_len.

        If method_param_path is set, includes method subdirectory:
        output_root/bundles/<hash>/<METHOD>/

        Otherwise uses flat structure:
        output_root/bundles/<hash>/

        Args:
            x: Parameter vector

        Returns:
            Path to bundle directory, or None if output_root not set
        """
        if not self.output_root:
            return None

        hash_str = self._hash_param_vec(x)

        # Build base bundle path
        if self.method_param_path:
            # Method-aware mode: output_root/bundles/<hash>/<METHOD>/
            bundle_base = self.output_root / "bundles" / hash_str

            # Extract method value from parameter vector
            param_dict = self.unpack(x)
            method_value = param_dict.get(self.method_param_path, "default")

            # Sanitize method value for use as directory name
            if isinstance(method_value, str):
                method_dir = method_value.replace("/", "_").replace("\\", "_")
            else:
                method_dir = str(method_value).replace("/", "_").replace("\\", "_")

            bundle_path = bundle_base / method_dir
        else:
            # Method-less mode: output_root/bundles/<hash>/
            bundle_path = self.output_root / "bundles" / hash_str

        # Check for MPI rank collision potential and append rank if needed
        """ 
        if HAS_MPI:
            try:
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                size = comm.Get_size()

                # If running under MPI with multiple ranks, check for potential collisions.
                # Note: This is a local check, so concurrent ranks might both see the path
                # as non-existent and try to create it. This is generally fine since:
                # 1. Parameter vectors are typically scattered uniquely across ranks
                # 2. Filesystem operations usually handle concurrent directory creation
                # For absolute safety with identical parameters across ranks, consider
                # adding rank suffix unconditionally when size > 1.
                if size > 1:
                    # Check if the base path would exist (potential collision)
                    if bundle_path.exists():
                        # In method-aware mode, append rank to method dir
                        if self.method_param_path:
                            bundle_path = bundle_path.parent / f"{method_dir}_r{rank}"
                        else:
                            # In flat mode, append rank to hash
                            bundle_path = self.output_root / "bundles" / f"{hash_str}_r{rank}"
            except:
                # If MPI detection fails, proceed without rank suffix
                pass
            """

        return bundle_path
    

    def _maybe_load_bundle(
        self,
        path: Path,
        cfg_override: Optional[Dict[str, Any]] = None,
        periods_to_load: Optional[List[int]] = None,
        stages_to_load: Optional[Dict[int, List[str]]] = None,
    ) -> Optional[Any]:
        """
        Wrapper around load_circuit catching FileNotFoundError.

        Args:
            path: Path to bundle directory
            cfg_override: Optional configuration to override loaded configs
            periods_to_load: Optional list of period indices to load
            stages_to_load: Optional dict mapping period indices to stage names

        Returns:
            ModelCircuit if loaded successfully, None otherwise
        """
        if not HAS_IO or not path or not path.exists():
            return None

        try:
            model = load_circuit(
                path, 
                cfg_override=cfg_override,
                periods_to_load=periods_to_load,
                stages_to_load=stages_to_load
            )
            return model
        except FileNotFoundError:
            return None
        except Exception as e:
            warnings.warn(f"Failed to load bundle from {path}: {e}")
            # Delete broken bundle to avoid hitting same error repeatedly
            try:
                import shutil

                if path.exists():
                    shutil.rmtree(path)
                    warnings.warn(f"Removed broken bundle directory: {path}")
            except Exception as cleanup_error:
                warnings.warn(f"Failed to clean up broken bundle: {cleanup_error}")
            return None

    def pack(self, d: Dict[str, Any]) -> np.ndarray:
        """
        Pack a dictionary of parameter values into a numpy array.

        Args:
            d: Dictionary mapping parameter paths to values

        Returns:
            Parameter vector with dtype=object for supporting mixed types

        Raises:
            ValueError: If a parameter path in d is not in param_paths
        """
        # Create array with object dtype to support mixed types
        x = np.empty(len(self.param_paths), dtype=object)

        # Fill array based on parameter paths order
        for i, path in enumerate(self.param_paths):
            if path in d:
                x[i] = d[path]
            else:
                raise ValueError(f"Parameter path '{path}' not found in input dictionary")

        return x

    def unpack(self, x: np.ndarray) -> Dict[str, Any]:
        """
        Unpack a parameter vector into a dictionary.

        Args:
            x: Parameter vector

        Returns:
            Dictionary mapping parameter paths to values

        Raises:
            ValueError: If the parameter vector length doesn't match param_paths
        """
        if len(x) != len(self.param_paths):
            raise ValueError(
                f"Parameter vector length ({len(x)}) doesn't match "
                f"number of parameters ({len(self.param_paths)})"
            )

        result = {}
        for i, path in enumerate(self.param_paths):
            result[path] = x[i]

        return result

    def patch_config(self, param_vec: np.ndarray) -> Dict[str, Any]:
        """
        Create a patched configuration by applying parameter values.

        Args:
            param_vec: Parameter vector of values to patch into the configuration

        Returns:
            A deep copy of the configuration with parameters patched

        Raises:
            ValueError: If the parameter vector length doesn't match param_paths
        """
        if len(param_vec) != len(self.param_paths):
            raise ValueError(
                f"Parameter vector length ({len(param_vec)}) doesn't match "
                f"number of parameters ({len(self.param_paths)})"
            )

        # Deep copy to avoid modifying the original
        cfg = copy.deepcopy(self.base_cfg)

        # Apply each parameter value
        for i, path in enumerate(self.param_paths):
            set_deep(cfg, path, param_vec[i])

        return cfg

    def run(                     # noqa: C901
        self, x: np.ndarray, *,
        return_model: bool = False,
        force_load: bool = False,
        save_bundle: Optional[bool] = None,
        rank: int = 0,
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Any]]:

        cfg         = self.patch_config(x)
        bundle_path = self._bundle_path(x)
        want_save   = self.save_by_default if save_bundle is None else save_bundle
        recorder    = RunRecorder()
        force_load  = self.load_if_exists 

        # ------------------------------------------------------------
        # 1. LOAD (root of comm_solver only)
        # ------------------------------------------------------------
        model = None
        if force_load and rank == 0:
            # Check if we have loading requirements stored on the runner
            periods_to_load = getattr(self, 'periods_to_load', None)
            stages_to_load = getattr(self, 'stages_to_load', None)
            
            model = self._maybe_load_bundle(
                bundle_path, 
                cfg_override=cfg,
                periods_to_load=periods_to_load,
                stages_to_load=stages_to_load
            )
            if model is None:
                raise FileNotFoundError(
                    f"[runner] bundle {bundle_path} not found or could not be loaded "
                    f"for parameter vector {x}. Aborting because fallback is disabled."
                )


        # ------------------------------------------------------------
        # 2. SOLVE (only if *not* in force-load mode)
        # ------------------------------------------------------------
        if not force_load:
            model = self.model_factory(cfg)
            self.solver(model, recorder=recorder)
            if self.simulator is not None:
                self.simulator(model, recorder=recorder)

            if want_save and rank == 0:
                bundle_path.parent.mkdir(parents=True, exist_ok=True)
                save_circuit(model, bundle_path.parent, self.base_cfg,
                            model_id=bundle_path.name)

        # ------------------------------------------------------------
        # 3. METRICS (root only)
        # ------------------------------------------------------------
        metrics: Dict[str, Any] = {}
        if rank == 0:
            # Import reference cache management
            try:
                from dynx.runner.reference_cache import release_strong_references
                has_ref_cache = True
            except ImportError:
                has_ref_cache = False
                
            # Pre-compute metric signatures for efficiency
            import inspect

            metric_signatures = {}
            for name, fn in self.metric_fns.items():
                try:
                    sig = inspect.signature(fn)
                    metric_signatures[name] = sig.parameters
                except Exception:
                    metric_signatures[name] = None

            for name, fn in self.metric_fns.items():
                try:
                    # Check if metric accepts keyword arguments
                    params = metric_signatures.get(name)
                    if params and "_runner" in params and "_x" in params:
                        # New-style metric with context
                        metrics[name] = fn(model, _runner=self, _x=x)
                    else:
                        # Legacy metric that only accepts model
                        metrics[name] = fn(model)

                    # Add metrics from recorder
                    metrics.update(recorder.metrics)

                except Exception as e:
                    import warnings

                    warnings.warn(f"Metric {name} failed with new signature, trying legacy: {e}")
                    try:
                        metrics[name] = fn(model)
                    except Exception as e2:
                        warnings.warn(f"Metric {name} failed completely: {e2}")
                        metrics[name] = np.nan
            
            # Release reference model cache after all metrics computed
            if has_ref_cache:
                release_strong_references()

        # ------------------------------------------------------------
        return (metrics, model) if return_model else (metrics, None)

def mpi_map(
    runner: "CircuitRunner",
    xs: np.ndarray,
    *,
    return_models: bool = False,
    mpi: bool = False,
    comm: Optional["MPI.Comm"] = None,
    comm_solver: Optional["MPI.Comm"] = None,
) -> Tuple[pd.DataFrame, List[Any]]:
    """
    Map ``runner.run`` over a design matrix.

    Parameters
    ----------
    runner : CircuitRunner
    xs : np.ndarray
        2-D design matrix (n_rows × n_params).
    return_models : bool, default False
        Return solved model objects *in single-process mode*.
        Under MPI they remain on the worker ranks; root receives ``[]``.
    mpi : bool, default True
        Attempt to use MPI if available (to split across runs)
    comm : mpi4py.MPI.Comm, optional
        Communicator to use to split parameters. 
    comm_solver : mpi4py.MPI.Comm, optional
        Communicator to use to split solver within a run. 

    Returns
    -------
    tuple
        ``(df, models)`` where

        * **df**   : pandas.DataFrame with parameters and metrics  
        * **models** : list of model objects (see above rules)
    """
    if xs.ndim != 2:
        raise ValueError("`xs` must be a 2-D array")

    # ------------------------------------------------------------------
    # Detect / configure MPI
    # ------------------------------------------------------------------
    mpi_enabled = False
    
    if mpi and comm is not None:
        mpi_enabled = comm.Get_size() > 1
    else:
        comm = None  # ensure serial code below sees `comm is None`

    rank = comm.Get_rank() if mpi_enabled else 0
    size = comm.Get_size() if mpi_enabled else 1

    rank_solver = comm_solver.Get_rank() if comm_solver is not None else 0
    size_solver = comm_solver.Get_size() if comm_solver is not None else 1

    # ------------------------------------------------------------------
    # Scatter rows
    # ------------------------------------------------------------------
    if mpi_enabled:
        chunks = np.array_split(xs, size) if rank == 0 else None
        chunk = comm.scatter(chunks, root=0)
    else:
        chunk = xs

    # ------------------------------------------------------------------
    # Local computation
    # ------------------------------------------------------------------
    local_metrics: List[dict] = []
    local_models: List[Any] = []

    for row in chunk:
        if return_models:
            metrics, model = runner.run(row, return_model=True, rank=rank_solver)
            local_models.append(model)
        else:
            metrics, model = runner.run(row, rank = rank_solver)

        # JSON-encode any non-scalar metric values
        for k, v in list(metrics.items()):
            if not np.isscalar(v) and not isinstance(v, str):
                metrics[k] = json.dumps(v)

        local_metrics.append(metrics)

    # Build local DataFrame
    param_df = pd.DataFrame(chunk, columns=runner.param_paths)
    metrics_df = pd.DataFrame(local_metrics)
    local_df = pd.concat([param_df, metrics_df], axis=1)

    # ------------------------------------------------------------------
    # Gather to root (MPI) or finalise (serial)
    # ------------------------------------------------------------------
    if mpi_enabled:
        all_dfs = comm.gather(local_df, root=0)

        if rank == 0:
            df = pd.concat(all_dfs, ignore_index=True)
            _write_design_matrix_csv(runner, xs)
            return df, []  # models live on workers
        else:
            return None  # type: ignore[return-value]

    # ---------- single-process return ----------
    if rank_solver == 0:
        _write_design_matrix_csv(runner, xs)
    
    return local_df, (local_models if return_models else [])



def _write_design_matrix_csv(runner: CircuitRunner, xs: np.ndarray) -> None:
    """
    Write design matrix CSV file with parameter hash and bundle directory mappings.

    Args:
        runner: CircuitRunner instance
        xs: Full design matrix (2D array)
    """
    if not runner.output_root:
        return  # No output directory configured

    try:
        # Create design matrix DataFrame
        df_design = pd.DataFrame(xs, columns=runner.param_paths)
        df_design["param_hash"] = [runner._hash_param_vec(row) for row in xs]

        # Get relative bundle paths
        bundle_dirs = []
        for row in xs:
            bundle_path = runner._bundle_path(row)
            if bundle_path:
                # Get path relative to bundles directory
                try:
                    rel_path = bundle_path.relative_to(runner.output_root / "bundles")
                    bundle_dirs.append(str(rel_path))
                except ValueError:
                    # If not relative to bundles, use full name
                    bundle_dirs.append(bundle_path.name)
            else:
                bundle_dirs.append("")

        df_design["bundle_dir"] = bundle_dirs

        # Output file path
        output_path = Path(runner.output_root) / "design_matrix.csv"

        # Append to existing file if it exists, avoiding duplicates (last-wins policy)
        if output_path.exists():
            try:
                existing_df = pd.read_csv(output_path)
                # Combine and remove duplicates based on param_hash, keeping last occurrence
                combined_df = pd.concat([existing_df, df_design], ignore_index=True)
                df_design = combined_df.drop_duplicates(subset=["param_hash"], keep="last")
            except Exception as e:
                warnings.warn(f"Failed to read existing design matrix: {e}, overwriting")

        # Write the updated design matrix
        df_design.to_csv(output_path, index=False)

    except Exception as e:
        warnings.warn(f"Failed to write design matrix CSV: {e}")


def plot_metrics(
    df: pd.DataFrame,
    metric_col: str,
    param_col: str,
    second_param_col: Optional[str] = None,
) -> None:
    """Plot metrics from a parameter sweep."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib import cm
    except ImportError:
        warnings.warn("matplotlib is required for plotting")
        return

    if second_param_col is None:
        # 1D plot
        plt.figure(figsize=(10, 6))
        plt.plot(df[param_col], df[metric_col], "o-")
        plt.xlabel(param_col)
        plt.ylabel(metric_col)
        plt.grid(True)
        plt.title(f"{metric_col} vs {param_col}")
    else:
        # 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        x = df[param_col].values
        y = df[second_param_col].values
        z = df[metric_col].values

        surf = ax.plot_trisurf(x, y, z, cmap=cm.viridis, linewidth=0.2, alpha=0.8)

        ax.set_xlabel(param_col)
        ax.set_ylabel(second_param_col)
        ax.set_zlabel(metric_col)

        plt.colorbar(surf)
        plt.title(f"{metric_col} vs {param_col} and {second_param_col}")

    plt.tight_layout()
    plt.show()


def plot_errors(df: pd.DataFrame, error_col: str, param_col: str) -> None:
    """Plot errors from a parameter sweep."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn("matplotlib is required for plotting")
        return

    plt.figure(figsize=(10, 6))
    plt.semilogy(df[param_col], df[error_col], "o-")
    plt.xlabel(param_col)
    plt.ylabel(error_col)
    plt.grid(True)
    plt.title(f"{error_col} vs {param_col}")
    plt.tight_layout()
    plt.show()
