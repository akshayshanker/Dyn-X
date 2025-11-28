# Change Log
## [Unreleased] - 2025-07-31

### Added
- **Selective Model Loading**:
  - Added `periods_to_load` parameter to `load_circuit()` for loading specific periods only
  - Added `stages_to_load` parameter to `load_circuit()` for loading specific stages per period
  - Updated `CircuitRunner._maybe_load_bundle()` to pass through selective loading parameters
  - CircuitRunner now checks for `periods_to_load` and `stages_to_load` attributes on runner instance

### Changed
- **Enhanced load_circuit() Signature**:
  - Function now accepts optional `periods_to_load: List[int]` parameter
  - Function now accepts optional `stages_to_load: Dict[int, List[str]]` parameter
  - Loading loop skips periods not in `periods_to_load` when specified
  - Loading loop skips stages not in `stages_to_load` for each period when specified
- **Enhanced load_reference_model() Function**:
  - Now accepts optional `metric_requirements` parameter for selective loading
  - Supports period and stage filtering when loading reference models for metrics
  - Enables metric-specific loading to reduce I/O overhead

### Improved
- **Performance for Large Models**:
  - Selective loading can reduce I/O by up to 76% for models with many periods
  - Memory usage reduced by only loading required data
  - Particularly beneficial for metric calculations that only need subset of periods
  - Example: Euler error calculation only needs periods 0-1 of a 5-period model

### Technical Details
- **Implementation**:
  - Modified `dynx/stagecraft/io.py`: Added filtering logic in `load_circuit()`
  - Modified `dynx/runner/circuit_runner.py`: Updated `_maybe_load_bundle()` and `run()`
  - Modified `dynx/runner/reference_utils.py`: Enhanced `load_reference_model()` with metric requirements
  - Backward compatible - existing code works without modification
  - Debug logging added for skipped periods/stages

## [0.1.8.dev6] - 2025-01-02

### Added
- **Reference Model Deviation Metrics**:
  - Added `method_param_path` parameter to CircuitRunner for method-aware bundle organization
  - Created new bundle directory structure: `output_root/bundles/<hash>/<METHOD>/`
  - Implemented reference model loading utilities in `dynx.runner.reference_utils`
  - Added deviation metric functions in `dynx.runner.metrics.deviations`:
    - L2 and L∞ norms for consumption (`dev_c_L2`, `dev_c_Linf`)
    - L2 and L∞ norms for assets (`dev_a_L2`, `dev_a_Linf`)
    - L2 and L∞ norms for value functions (`dev_v_L2`, `dev_v_Linf`)
    - L2 and L∞ norms for policy functions (`dev_pol_L2`, `dev_pol_Linf`)
  - Added `make_policy_dev_metric` factory function for creating custom deviation metrics
  - Enhanced metric invocation to pass `_runner` and `_x` arguments for context-aware metrics

### Changed
- **Bundle Path Structure**:
  - Modified `_bundle_path` to include method subdirectories when `method_param_path` is set
  - Method-less projects (with `method_param_path=None`) continue to use flat bundle structure
  - Bundle paths now follow: `bundles/<param_hash>/<method_name>/` in method-aware mode
- **Metric Function Signature**:
  - Metric functions can now accept optional `_runner` and `_x` keyword arguments
  - Backward compatibility maintained through try/except fallback for legacy metrics

### Improved
- **Reference Model Workflow**:
  - Users can solve high-accuracy reference models once and compare multiple fast methods
  - Deviation metrics automatically load reference solutions when available
  - Metrics return `NaN` when reference model is missing (non-breaking behavior)
  - Default reference method is `VFI_HDGRID` (configurable via `DEFAULT_REF_METHOD`)

### Documentation
- **Added Unit Tests**:
  - Comprehensive test suite in `tests/unit/test_runner_deviation.py`
  - Tests for method-aware bundle paths, deviation calculations, and backward compatibility
  - Tests for method-less mode ensuring flat bundle structure is preserved

### Technical Implementation
- **New Modules**:
  - `dynx/runner/reference_utils.py`: Functions for locating and loading reference models
  - `dynx/runner/metrics/deviations.py`: Policy deviation metric implementations
  - `dynx/runner/metrics/__init__.py`: Package initialization with metric exports
- **Helper Functions**:
  - `ref_bundle_path()`: Compute path to reference bundle for given parameters
  - `load_reference_model()`: Load reference model if it exists

### Migration Notes
- Existing code continues to work unchanged (backward compatible)
- To use deviation metrics, add `method_param_path` to CircuitRunner (default: `"master.methods.upper_envelope"`)
- Add deviation metrics to `metric_fns` dictionary: `{"dev_c_L2": dev_c_L2, ...}`
- Reference models should be solved with `DEFAULT_REF_METHOD` (typically `"VFI_HDGRID"`)

### Usage Example
```python
from dynx.runner import CircuitRunner
from dynx.runner.metrics import dev_c_L2, dev_c_Linf

# Configure runner with method awareness
runner = CircuitRunner(
    base_cfg=cfg,
    param_paths=["beta", "sigma", "master.methods.upper_envelope"],
    model_factory=make_model,
    solver=solve,
    metric_fns={"dev_c_L2": dev_c_L2, "dev_c_Linf": dev_c_Linf},
    output_root="results/",
    save_by_default=True,
    method_param_path="master.methods.upper_envelope"
)

# Solve reference (once)
x_ref = runner.pack({"beta": 0.96, "sigma": 2.0, "master.methods.upper_envelope": "VFI_HDGRID"})
runner.run(x_ref)

# Compare fast methods
x_fast = runner.pack({"beta": 0.96, "sigma": 2.0, "master.methods.upper_envelope": "FUES"})
metrics = runner.run(x_fast)
print(f"L2 deviation: {metrics['dev_c_L2']}")
```

## [0.1.8.dev5] - 2024-12-27

### Added
- **Disk-Based Save/Load Support for CircuitRunner**:
  - Implemented complete bundle management system allowing each `runner.run(x)` invocation to save solved ModelCircuits to disk and load existing bundles to skip re-solving
  - **New CircuitRunner Parameters**:
    - `output_root`: Parent directory for bundle storage (None disables saving unless overridden)
    - `bundle_prefix`: First part of bundle folder names (default: "run")
    - `save_by_default`: Auto-save every solved model (default: False)
    - `load_if_exists`: Auto-load existing bundles instead of solving (default: False)
    - `hash_len`: Number of hex chars from MD5 hash for directory naming (default: 8)
    - `config_src`: Source configuration to pass to save_circuit (preserves original config format)
  - **Enhanced run() Method**:
    - Added `save_model` parameter to override `save_by_default` for individual calls
    - Method signature now uses keyword-only arguments: `run(x, *, return_model=False, save_model=None)`
  - **Reserved Design Matrix Parameter**:
    - `__runner.mode` parameter accepts "solve" or "load" to override all flags for specific rows
    - Extracted from parameter vector and controls solve/load behavior per invocation
  - **Bundle Structure**:
    - Directory naming: `<output_root>/<bundle_prefix>_<hash>` where hash is first `hash_len` chars of MD5(parameter vector)
    - Uses existing `save_circuit()` and `load_circuit()` from `dynx.stagecraft.io`
    - Enhanced manifest with bundle metadata: hash, prefix, saved_by_rank, and complete parameter dictionary
  - **Design Matrix CSV Export**:
    - Automatic creation of `<output_root>/design_matrix.csv` after parameter sweeps
    - Contains columns for all parameter paths plus `param_hash` and `bundle_dir`
    - Supports incremental updates with duplicate detection for multiple sweeps
    - Single-write semantics: only rank 0 writes in MPI mode
  - **MPI Compatibility**:
    - Each rank uses same hash function for deterministic bundle paths
    - Automatic rank collision detection: appends `_r{rank}` to prefix when filesystem collision detected
    - Workers can save bundles locally while rank 0 aggregates results
    - Full compatibility with existing `mpi_map()` function

### Changed
- **CircuitRunner API Evolution**:
  - `run()` method now supports keyword-only arguments for clarity
  - Enhanced parameter unpacking to handle reserved `__runner.mode` parameter
  - Improved MPI detection logic in `mpi_map()` to avoid variable shadowing
  - Added graceful fallbacks when `dynx.stagecraft.io` is not available
  - Fixed caching logic to always compute hash key but only use cache when enabled

### Improved
- **Bundle Management Features**:
  - Deterministic bundle naming based on parameter vector hashing
  - Smart load/save logic with multiple override mechanisms
  - Comprehensive manifest generation including all run parameters
  - Robust error handling for save/load operations with helpful warnings
  - Automatic directory creation and collision avoidance
  - **Cross-Machine Compatibility**: Automatic stage re-compilation after loading bundles
  - **Broken Bundle Cleanup**: Automatic removal of corrupted bundles to prevent repeated failures
- **Config Source Handling**:
  - Support for preserving original config source format (files, directories, dicts)
  - Fallback to synthetic container creation when config_src not provided
  - Better compatibility with various configuration input formats

### Documentation
- **New Examples**:
  - Updated `examples/circuit_runner_saving.py` with design matrix CSV demonstration
  - Comprehensive test suite in `tests/runner/test_circuit_runner_save_load.py`
  - Added tests for design matrix creation, deduplication, and error handling
  - Examples cover basic save/load, mode parameter usage, and parameter sweeps with bundle management

### Technical Implementation
- **Helper Methods**:
  - `_bundle_path(x)`: Generate bundle directory paths with collision detection
  - `_hash_param_vec(x)`: Create deterministic hashes of parameter vectors  
  - `_maybe_load_bundle(path)`: Safe wrapper around `load_circuit()` with error handling and cleanup
  - `_write_design_matrix_csv()`: Create and maintain design matrix CSV files
- **Config Container Creation**: Automatically structures base_cfg into proper container format for `save_circuit()`
- **Manifest Enhancement**: Extends base manifest with bundle-specific metadata and parameter tracking

### Fixed
- **RunRecorder Interface**: Updated all examples and tests to use `recorder.add()` instead of deprecated `recorder.record()`
- **Cache Key Logic**: Fixed issue where cache key was computed conditionally but used unconditionally
- **MPI Variable Shadowing**: Resolved shadowing of module-level `HAS_MPI` variable in `mpi_map()`
- **Bundle Loading**: Added stage re-compilation after loading for cross-machine compatibility
- **Error Recovery**: Improved error handling with automatic cleanup of broken bundles

### Migration Notes
- The `run()` method signature change requires keyword arguments for `return_model` and the new `save_model` parameters
- Bundle features are optional and disabled by default - no breaking changes for existing code
- When `dynx.stagecraft.io` is unavailable, bundle features are disabled with warnings
- MPI behavior remains unchanged; bundles provide additional persistence without affecting parallelization
- Design matrix CSV files are created automatically and can be used for post-experiment analysis

### Usage Example
```python
runner = CircuitRunner(
    base_cfg=cfg,
    param_paths=["beta", "sigma"],
    model_factory=make_model,
    solver=solve,
    output_root="solutions/",
    bundle_prefix="housing",
    save_by_default=True,
    load_if_exists=True,
    config_src=original_config_path,  # Preserve original format
)

# Run parameter sweep - creates bundles and design_matrix.csv
df = mpi_map(runner, design_matrix, return_models=False)

# Analyze results using design matrix
import pandas as pd
design_df = pd.read_csv("solutions/design_matrix.csv")
print(f"Completed {len(design_df)} parameter combinations")
for _, row in design_df.iterrows():
    print(f"  {row['param_hash']}: {row['bundle_dir']}")
```

## [0.1.8.dev4] - 2024-12-26

### Fixed
- **Solution Object Re-hydration**:
  - Fixed issue where loaded Solution objects were plain dictionaries instead of proper Solution instances
  - `load_circuit` now detects when a pickled object is a Solution dictionary and automatically converts it back to a Solution object
  - This ensures downstream code that expects Solution attributes (like `.EGM.unrefined`) works correctly
  - Regular dictionaries that don't match the Solution structure are preserved as-is
- **Branch Dictionary Handling**:
  - Fixed serialization of branch dictionaries containing Solution objects (e.g., TENU.cntn.sol with `{"from_owner": <Solution>, "from_renter": <Solution>}`)
  - Previously failed with "not pickle-able" error due to numba typed.Dict objects inside Solutions
  - Now automatically converts each Solution to dict before pickling and re-hydrates on load
  - Prevents zero-byte sol.pkl files and subsequent EOFError when loading

### Changed
- **Enhanced Config Saving**:
  - Added `_dump_yaml` helper for cleaner YAML writing
  - `_copy_configs` now supports canonical container dictionaries with `{"master", "stages", "connections"}` structure
  - Improved organization when saving stage configurations to the `stages/` subdirectory

### Technical Details
- Solution detection uses key-set check: `{"policy", "EGM", "timing"}` to identify Solution dictionaries
- Branch dictionary detection identifies dicts containing Solution objects and serializes each branch separately
- Re-hydration applies to both `sol.pkl` and `sim.pkl` files
- Uses `Solution.from_dict()` to properly reconstruct the object with all its attributes

## [0.1.8.dev3] - 2024-12-26

### Added
- **Folder-Based Config Loading**:
  - Added `load_config` helper function in `dynx.stagecraft.io` to load configurations from a structured directory
  - Expected folder structure:
    ```
    config_dir/
    ├── master.yml
    ├── stages/
    │   ├── stage1.yml
    │   └── stage2.yaml    # Both .yml and .yaml extensions supported
    └── connections.yml
    ```
  - Returns a dictionary with keys "master", "stages", and "connections"
  - Supports both `.yml` and `.yaml` file extensions for stage configurations
  - Stage names are automatically uppercased to ensure consistency with Stage object names
  
### Changed
- **Refactored `load_circuit`**:
  - Now accepts only a directory path (no more file/zip arguments)
  - Uses the new `load_config` helper internally
  - Added `restore_data` parameter (default True) to control whether solution/distribution data is restored
  - Improved error messages for missing files and directories
- **Stage Name Normalization**:
  - Stage configuration keys are now automatically uppercased when loading (e.g., `tenu.yml` → `"TENU"` key)
  - Ensures compatibility regardless of filename casing conventions

### Fixed
- **Import and Path Handling**:
  - Added missing `argparse` import for CLI functionality
  - Wrapped paths in `str()` for FileNotFoundError messages to avoid Python 3.8 compatibility issues
  - Confirmed `.yaml` extension support in `_copy_configs` (already present)
  
### Improved
- **Test Suite Updates**:
  - Updated `tests/saver/test_saver.py` to use folder-based configurations
  - Added dedicated test for `load_config` function
  - Added test for verifying folder structure after save
  - Added tests for `.yaml` file extension support

### Migration Notes
* If you were passing individual YAML files to `load_circuit`, update to pass only the directory path
* Config loading now expects the standard folder structure with `master.yml`, `connections.yml`, and `stages/` subdirectory
* The `load_config` function can be used standalone to load configurations for custom workflows
* Stage configuration files can use either `.yml` or `.yaml` extensions

## [0.1.8.dev2] - 2024-12-26

### Changed
- **Module Renaming**:
  - Renamed `config_loader` module to `makemod` for better clarity and naming consistency
  - Updated all imports throughout the codebase from `dynx.stagecraft.config_loader` to `dynx.stagecraft.makemod`
  - Updated test files to use the new module name
  - Fixed `saver.py` to properly load YAML configurations and pass them as dictionaries to `initialize_model_Circuit`
  - Updated saver.py to io.py for clarity. 

### Migration Notes
* Update all imports: Change `from dynx.stagecraft.config_loader import ...` to `from dynx.stagecraft.makemod import ...`
* The functionality remains the same - only the module name has changed
* The main function `initialize_model_Circuit` and helper functions like `compile_all_stages` retain their original names

Refactor
## [0.1.8.dev1] - 2025-05-16

### Added
* New `Solution` container class in `stagecraft.solmaker` for storing stage solutions
* Numba-compatible solution storage with support for arbitrary-dimensional arrays
* Dual access pattern (attribute and dictionary-style) for all solution fields
* Support for nested policy dictionaries and EGM layer storage
* Save/load functionality using NPZ + JSON format
* Comprehensive unit tests for the Solution container



### Migration Notes
* Existing code accessing `stage.dcsn.sol["policy"]` should now use `stage.dcsn.sol.policy["c"]` for consumption policy
* Housing/service policies are now accessed as `sol.policy["H"]` and `sol.policy["S"]` respectively
* EGM grids are accessed via `sol.EGM.refined.e` instead of `sol["EGM"]["refined"]["e"]`
* The Solution object can be converted to/from plain dicts using `as_dict()` and `from_dict()` methods

## [0.1.8.dev0] - 2025-05-16

* MPI circuit runner does not pickle model classes and only gathers and returns results from rank 0.


## [0.1.8.dev0] - 2025-05-16

* Public dev version in GitHub.

## [0.1.7.1.3.dev0] - 2025-05-16

⚠️ **Experimental Development Build**  
The **entire Dyn-X / ModCraft codebase is under active development and should be considered *experimental***.  
Breaking changes can land at any time; if you need a fixed reference point pin **a specific commit hash** or use an explicit dev pre-release such as `dynx==1.7.1.dev3`.

### Added
* **Sampler Layer Overhaul** – see prompts `prompt_v1.7.1.1-3.md`
  * `dynx.runner.sampler` now ships:
    * `BaseSampler`, `MVNormSampler`, `FullGridSampler`, `LatinHypercubeSampler`, `FixedSampler`.
    * `build_design()` helper that produces a single `dtype=object` design matrix and accompanying `info` dict.
  * New helper `_is_categorical()` to detect `enum` / `values` meta entries.
* **CircuitRunner Quality-of-Life**
  * `pack()` / `unpack()` helpers for converting between dict↔array with mixed dtypes.
  * `plot_metrics()` / `plot_errors()` convenience plots.
* **Test Suites** under `tests/runner/` & `tests/unit/runner/`
  * `test_sampler.py`, `test_build_design.py`, `test_mvn_bounds.py` for sampler layer.
  * `test_circuit_runner.py` integration smoke-tests.

### Changed
* `CircuitRunner`
  * Switched from `pickle.dumps` → `copy.deepcopy` for config cloning.
  * Improved cache-key hashing logic (handles `dtype=object`).
  * **Deprecated** `sampler` constructor arg – emits warning; will be removed in 0.1.7.
* `MVNormSampler` clips / resamples draws to honour numeric bounds when `meta` provided.
* `FullGridSampler` accepts categorical string grids.
* `LatinHypercubeSampler` now yields `dtype=float` but overall `build_design` promotes to `object` when categoricals present.
* `build_design()` signature changed: `(param_paths, samplers, Ns, meta, seed=None)` returning `(xs, info)`.
* Package exports: `dynx.runner.__init__` now re-exports all sampler classes and helpers.
* **Code Style** – Black 88-char line length enforced across modified files.

### Removed
* Legacy `enum_specs` integer mapping logic (categoricals are now inserted literally).

### Fixed
* Sampler dtype handling for mixed categorical/continuous designs.
* Assorted unit-test failures arising from interface changes.

### Migration
1. Replace any `CircuitRunner(..., sampler=<something>)` with standalone sampler calls:
   ```python
   xs, _ = build_design(param_paths, [MySampler], [N], meta)
   runner = CircuitRunner(base_cfg, param_paths, model_factory, solver)
   ```
2. Update imports:
   ```python
   from dynx.runner import MVNormSampler, build_design, CircuitRunner
   ```

---

## [0.1.6.12.dev0] - 2023-11-17

### Changed
- **Package API Simplification and Restructuring**:
  - Removed deprecated `dynx_runner` compatibility shim in favor of direct imports from `dynx.runner`
  - Moved all runner functionality directly into `dynx.runner` module
  - Eliminated external `dynx_runner` package access entirely
  - Updated all internal imports and cross-references to use the new direct import pattern
  - This is a breaking change for code that directly imports from `dynx_runner`
  - All functionality is preserved through the `dynx.runner` module

### Added
- **Improved Code Organization**:
  - Centralized all core functionality within the main `dynx` package
  - Enhanced module structure with clearer separation of concerns
  - Better compliance with Python packaging best practices

### Fixed
- **Test Suite Restructuring**:
  - Reorganized tests into a clear hierarchy with `unit`, `integration`, `regression`, and `data` directories
  - Moved runner tests into `tests/unit/runner/` directory
  - Updated test imports to use the new module structure
  - Fixed import-related test failures in regression tests
  - Ensured all tests pass with the new package structure

### Migration
- Replace imports from `dynx_runner` with imports from `dynx.runner`:
  - Change `from dynx_runner import CircuitRunner` to `from dynx.runner import CircuitRunner`
  - Change `import dynx_runner` to `import dynx.runner`
  - Change `from dynx_runner.circuit_runner import mpi_map` to `from dynx.runner import mpi_map`
  - Update any other imports to use the corresponding paths in the `dynx.runner` module

### Rationale
- **Simplification and Integration**:
  - The separate `dynx_runner` package created unnecessary complexity
  - Direct integration into the main package improves discoverability and usability
  - Single-import pattern makes the API more consistent and intuitive
  - Reduces maintenance burden by eliminating duplicate code paths
  - This change aligns with the project's goal of providing a cohesive, well-structured framework

## [0.1.6.11.dev0] - 2023-11-16

### Added
- **Implementation of Variable Resolution Spec**:
  - Started implementation of variable scoping rules based on prompt_v_1.6.11.md
  - Fixed issues with mover.py to properly handle nested attribute references 
  - Enhanced state space resolution in the whisperer.py module
  - Added proper error handling when attributes are missing in model structure
  - Improved initialization of attributes to prevent AttributeError exceptions

  YAML Configuration Parameter Resolution Highlights
  1. Manual Shock Process Support (v1.6.5)
  Implemented reference resolution for transition matrices and shock values from master configs
  Enhanced parameter resolution to handle nested references in matrices
  Added support for manually specified shock processes with parameter validation
  2. Master Config Handling Improvements (v1.6.8)
  Fixed critical issues when handling null master configurations
  Added proper null checks before accessing master_math_functions
  Enhanced robustness when processing function inheritance without a master configuration
  3. Parameter Resolution Enhancements (v1.6.9)
  Enhanced parameter resolution to prevent circular references
  Improved grid access with multiple fallback patterns
  Fixed model transfer to ensure grids are available in all perch models
  4. Variable Resolution Specification (v1.6.11)
  Started implementation of variable scoping rules based on prompt_v_1.6.11.md
  Fixed issues with mover.py to properly handle nested attribute references
  Enhanced state space resolution in the whisperer.py module
  Added proper error handling when attributes are missing in model structure
  Improved initialization of attributes to prevent AttributeError exceptions
  5. Earlier Significant Parameter Resolution Changes (v1.6.0)
  Implemented a consistent global reference rule: [name] for lookup vs literal values
  Added resolve_tag() helper for centralizing resolution of references
  Added unified error handling for references that can't be resolved
  Enhanced parameter resolution to handle the reference syntax consistently
  Implemented proper unpacking of reference values from lists
  6. Grid Generation and Parameter Resolution (v1.5.19)
  Added variable resolution for grid specifications from settings (state-specific and global)
  Enhanced grid parameters to support variable references from settings
  Fixed variable resolution in int_range style grid specifications
  7. Parameter Resolution and Error Handling (v1.5.17)
  Optimized parameter resolution with loop prevention and improved type handling
  Fixed parameter resolution for string references with maximum recursion depth
  Enhanced type checking and conversions for more robust numeric operations
  These improvements collectively demonstrate a focus on making parameter resolution more robust, flexible, and consistent throughout the codebase, with particular emphasis on handling references, preventing circular dependencies, and providing better error messages when resolution fails.

## [0.1.6.10.dev0] - 2023-11-15

### Added
- **Numerical Compilation Helper for Stages**:
  - Implemented `compile_all_stages` public helper function in config_loader.py
  - Added ability to compile all stages in a model with a single function call
  - Created robust error handling with per-stage try/except blocks
  - Added support for optional forcing of recompilation with `force=True` parameter
  - Implemented grid/mesh proxy attachment after numerical generation
  - Added comprehensive logging throughout the compilation process
  - Created status flag tracking (`compiled`) for stages that complete generation

### Improved
- **Model Visualization and Plotting**:
  - Enhanced plotting functions for economic models with standardized styling
  - Implemented direct model_circuit access in visualization functions
  - Added robust grid and solution data access with proper attribute checking
  - Created consistent styling for plots matching established economics papers
  - Improved error handling for missing or incomplete data in visualization
  - Enhanced visualization of policy functions across different parameter values
  - Added support for smooth visualization of policy functions with discontinuities
  - Fixed issues with blank plots by ensuring direct access to model data

### Documentation
- **Best Practices**:
  - Added visualization best practices addendum to the documentation
  - Created examples of direct model_circuit access patterns
  - Documented proper grid indexing for multi-dimensional state spaces
  - Added guidelines for consistent plot styling and formatting
  - Included examples of error handling for robust visualization code

## [0.1.6.9.dev0] - 2023-11-14

### Added
- **Enhanced Grid and Mesh Access**:
  - Implemented `_MeshProxy` class for transparent access to mesh grids in state spaces
  - Added conditional mesh proxy attachment based on grid existence
  - Added direct `.mesh` attribute to perch objects for accessing mesh grids
  - Created comprehensive test suite for grid and mesh access patterns
  - Added detailed documentation and cheat sheets for all access patterns
  - Ensured backward compatibility with existing code

### Improved
- **State Space Structure and Access**:
  - Fixed state space grid generation to properly handle multi-dimensional grids
  - Enhanced grid access with multiple fallback patterns for robust access
  - Standardized access patterns for both grid and mesh structures
  - Improved error messaging for grid/mesh access failures
  - Enhanced parameter resolution to prevent circular references
  - Fixed model transfer to ensure grids are available in all perch models

### Documentation
- **Access Pattern Documentation**:
  - Added `access_tests.md` cheat sheet with all grid/mesh access patterns
  - Created detailed examples for working with state space functions
  - Documented proper hierarchy for state space components
  - Added mesh grid creation and manipulation examples
  - Enhanced docstrings for all grid/mesh related classes and methods

## [0.1.6.8.dev0] - 2023-11-13

### Fixed
- **Improved Master Config Handling**:
  - Fixed critical issue in `build_stage` function when `master_config` is `None`
  - Added proper null checks before attempting to access `master_math_functions.items()`
  - Replaced ambiguous truthiness checks with explicit `is not None` comparisons
  - Enhanced robustness when processing function inheritance without a master configuration
  - Improved error handling in stage initialization process
  - Ensures stage configs can be loaded successfully without requiring a master config

## [0.1.6.5.dev0] - 2023-11-10

### Added
- **Manual Shock Process Support**:
  - Implemented support for manually specified shock processes in Heptapod-B
  - Added ability to directly specify transition matrices and shock values in YAML
  - Created `manual_shocks.py` module with comprehensive validation and parameter resolution
  - Added support for both DiscreteMarkov and IID shock processes with manual specification
  - Implemented reference resolution for transition matrices and shock values from master config
  - Added automatic stationary distribution calculation for manually specified Markov processes

### Changed
- **Enhanced Shock Generation**:
  - Modified shock generation to detect and handle the "manual" method
  - Added backward compatibility with existing algorithmic shock generation
  - Improved shock object storage and access with consistent pattern
  - Enhanced parameter resolution to handle nested references in matrices

### Documentation
- **New Examples and Tests**:
  - Added comprehensive test suite for manual shock specification
  - Created example scripts demonstrating manual shock creation and usage
  - Added YAML example configuration showing different manual shock patterns
  - Added visualization support for transition matrices
  - Ensured proper integration with the math/num separation in FunctionalProblem

## [0.1.6.4.dev0] - 2025-04-23

### Added
- **Refactored Config Loader with Phased Architecture**:
  - Implemented clear separation of phases for model building: validation, period creation, connections, registration
  - Added custom exception hierarchy (LoaderError, ConfigKeyError) for better error handling
  - Created utility functions (_as_int, _ensure_list, _resolve_period_indices) for robust data handling
  - Implemented comprehensive type hints for better IDE support and code safety
  - Added SVG output option for visualizations

### Changed
- **Improved Code Structure and Organization**:
  - Broke down monolithic functions into smaller, focused helpers with single responsibilities
  - Organized imports into logical sections (standard library, third-party, local)
  - Extracted constants for layouts, edge styles, and visualization defaults
  - Replaced all print statements with logger.info for deterministic logging
  - Enhanced error handling with more specific, context-rich error messages

### Improved
- **Connection Handling and Visualization**:
  - Enhanced connection creation with normalized dictionaries (_iter_intra_conn, _apply_intra_conn)
  - Improved inter-period connections with better source/target perch attribute management
  - Consolidated visualization code with consistent parameter structure
  - Added uniform generation of visualizations for all supported layouts
  - Enhanced readability of generated graphs with consistent styling

## [0.1.6.3.dev0] - 2025-04-22

### Added
- **YAML Configuration Structure Improvements**:
  - Implemented complete separation of concerns between stage configs, connections, and master configs
  - Added support for dedicated `connections.yml` file for defining model topology
  - Created pure function-based `initialize_model_Circuit` approach that accepts pre-loaded config dictionaries

### Changed
- **Two-Phase Model Construction Process**:
  - Implemented Phase 1: Creating all stages before establishing any connections
  - Implemented Phase 2: Establishing connections after all stages are created
  - Modified connection creation order: intra-period connections before periods are added to model
  - Enhanced parameter inheritance through proper propagation of master_config

### Improved
- **Connection Handling**:
  - Enhanced `determine_required_periods` to analyze connection configs for period requirements
  - Fixed `create_intra_period_connections` to support both dictionary and list formats
  - Improved `create_inter_period_connections` with support for period arrays and branch keys
  - Added transpose (backward) connection creation for inter-period relationships

### Fixed
- **Visualization and Error Handling**:
  - Corrected period access to use `model.periods_list` instead of invalid `model.periods` attribute
  - Fixed indentation errors in exception handling blocks
  - Added comprehensive logging throughout the model building process
  - Enhanced visualization with multiple formats (hierarchical, circular, spring layout)

## [0.1.6.2.dev0] - 2023-11-08

### Changed
- **API Consistency Improvements**:
  - Renamed `math["state"]` to `math["state_space"]` for consistency with the numerical representation
  - Renamed `parameters` attribute to `parameters_dict` in `FunctionalProblem` for clearer distinction between dictionary and attribute access
  - Updated all module files and examples to use the new consistent naming
  - Improved API consistency by aligning structure between mathematical and numerical representations

### Fixed
- **Grid Display and Access**:
  - Fixed grid display in examples to correctly handle state space structure
  - Improved debugging and structure inspection for state space objects
  - Enhanced error handling for grid access and display
  - Updated example files to use correct dictionary access patterns with the state_space structure

### Improved
- **Documentation and Testing**:
  - Updated class docstrings to better describe the accessible attributes
  - Enhanced test coverage for state space structure and grid access
  - Added comprehensive tests for numerical state space generation
  - Clarified access patterns for container-based attribute access
  - Improved consistency in property access patterns to avoid confusion

## [0.1.6.1.dev0] - 2025-05-15

### Added
- **Examples Directory Reorganization**:
  - Created structured directory hierarchy for improved organization:
    - `examples/heptapod_b/` for Heptapod-B specific examples
    - `examples/economic_models/` for economic model implementations 
    - `examples/workflows/` for workflow examples
    - Retained `examples/configs/` and `examples/circuitCraft/`
  - Added README.md files with detailed documentation for each major directory
  - Enhanced main examples README with comprehensive guides on parameter resolution and grid specifications
  - Added special documentation sections covering:
    - Parameter resolution syntax and approach
    - Grid specification options and best practices
    - Common workflow patterns and implementation strategies

### Improved
- **Documentation Structure**:
  - Added category-specific README.md files with relevant example descriptions
  - Implemented cross-referencing between related examples
  - Added directory-level documentation explaining purpose and contents
  - Created consistent file organization pattern across all example categories
  - Ensured proper Python package structure with __init__.py files in all subdirectories
  - Enhanced code comments for easier onboarding of new users
  - Added explicit links between examples and corresponding documentation sections

### Changed
- **Example Organization**:
  - Moved example files to their appropriate category directories
  - Grouped examples by functionality rather than by API or technique
  - Applied consistent naming conventions across example files
  - Structured economic models into domain-specific subdirectories
  - Updated import paths in all example files to reflect new directory structure
  - Ensured backward compatibility through appropriate __init__.py configuration
  - Added cross-references in code comments to related examples

## [0.1.6.0.dev0] - 2025-05-01

### Added
- **Streamlined YAML Grammar**:
  - Implemented a consistent global reference rule: `[name]` for lookup vs literal values
  - Added `resolve_tag()` helper for centralizing resolution of references across the framework
  - Created `tools/migrate_yaml_v15_to_v16.py` script for bulk-upgrading YAML files
  - Added unified error handling for references that can't be resolved

- **Enhanced Model API with Intuitive Attribute Access**:
  - Added attribute-style (dot notation) access to nested dictionaries
  - Implemented `_AttributeAccessWrapper` class to provide attribute access to dictionary items
  - Added `param` property for direct parameter access with `problem.param.beta` syntax
  - Maintained full backward compatibility with dictionary-style access
  - Added robust support for nested dictionaries with consistent behavior

### Changed
- **Simplified Configuration Structure**:
  - Eliminated redundant `grid_generation` field, using grid block's `type` directly
  - Consolidated all compilation switches into a single `compilation` field
  - Replaced `shock_distribution` and `shock_generation` with unified `shock_method`
  - Renamed `interpolation` to `interp` for consistency and brevity
  - Simplified multi-output function format by removing nested `inputs`/`outputs`/`exprs` structure
  - Streamlined grid parameters to use `points` consistently instead of `n`
  - Replaced `create_mesh` flag with more intuitive `no_mesh` toggle

- **Code Cleanup**:
  - Removed legacy properties `self.functions` and `self.states` that are no longer needed
  - Enhanced initialization process to apply attribute access after model is fully constructed
  - Updated dictionary access methods to maintain backward compatibility

### Improved
- **Code Organization**:
  - Enhanced `resolve_grid_type()` for more robust grid type resolution
  - Improved handling of lists in parameter values during grid generation
  - Added comprehensive deprecation warnings until v1.7 for backward compatibility
  - Enhanced parameter resolution to handle the reference syntax consistently
  - Implemented proper unpacking of reference values from lists
  - Updated all examples to use the new streamlined syntax

- **User Experience**:
  - More intuitive model component access with Python's natural dot notation
  - Better IDE support with autocompletion for model components
  - Simplified code with less verbose access patterns
  - Enhanced readability in examples and application code

### Documentation
- **Enhanced Configuration Guide**:
  - Added detailed descriptions of the reference vs literal rule
  - Updated all examples to demonstrate the new syntax
  - Created comprehensive migration guide in the spec document
  - Added detailed comments in examples explaining the v1.6 format changes

- **New API Documentation**:
  - Created new `prompt_v1.6.md` documenting the attribute access enhancements
  - Added usage examples for the new attribute access patterns
  - Included implementation details for developers
  - Documented backward compatibility measures

## [0.1.5.19.dev0] - 2025-04-23

### Added
- **Multi-Dimensional State Spaces & Manual Grid Specification**:
  - Implemented direct grid specification via `grid:` YAML key to bypass algorithmic generation
  - Added support for manual grid specification as either flat lists or dimension-mapped values
  - Created standardized grid-generation alias table mapping shorthand to canonical types
  - Added schema validation for grid types with clear required parameter documentation
  - Implemented type inference for manually specified grids
  - Added `create_mesh` flag to control tensor product creation for multi-dimensional spaces
  - Added variable resolution for grid specifications from settings (state-specific and global)
  - **New Examples**:
    - Added detailed age-asset lifecycle model example showing multi-dimensional state space
    - Demonstrated range specification syntax for both continuous and integer dimensions
    - Enhanced `ConsInd_multi.yml` example to showcase the new manual grid specification format

### Improved
- **Grid Generation System**:
  - Established canonical design principles: no example-level patching, accuracy-first validation
  - Simplified fallback mechanism with single default (`type = linspace`)
  - Enhanced parser to detect and validate manual grid specifications
  - Updated numeric builder to handle explicit grid definition cleanly
  - Made multi-dimensional grid specification more intuitive and aligned with NumPy capabilities
  - Improved performance by avoiding unnecessary mesh creation when manually specified
  - Enhanced grid parameters to support variable references from settings
  - Updated `load_consind_multi.py` example to display generated grid dimensions and ranges

### Fixed
- **Parameter Handling in Function Definitions**:
  - Clarified that model parameters should NOT be listed in the `inputs` array of functions
  - Ensured parameters are automatically baked in from the model's parameters section
  - Consolidated duplicate code from modified files into canonical implementations
  - Fixed variable resolution in int_range style grid specifications

### Documentation
- **Enhanced Grid Documentation**:
  - Added comprehensive grid types reference table
  - Created examples demonstrating each grid specification approach
  - Documented required keys for each grid type
  - Updated model structure documentation with explicit grid specification examples
  - Added detailed examples showing various grid specifications in `manual_grid_example.yml`
  - Added documentation on correct parameter usage in multi-output functions

## [0.1.5.18.dev0] - 2025-04-22

### Fixed
- **ConsInd Example and DEGM Whisperer**:
  - Fixed grid access path in whisperer.py to correctly use `stage.model.num["cntn"]["a_nxt"]` rather than `stage.cntn_to_dcsn.model.num["cntn"]["a_nxt"]`
  - Updated ConsInd.yml with proper numeric values for grid parameters to prevent type errors during grid generation
  - Improved robustness of the ConsInd.py example for running lifecycle consumption-savings models
  - Enhanced compatibility between the ModCraft framework and the DEGM model implementation

## [0.1.5.17.dev0] - 2025-04-21

### Improved
- **Code Quality Enhancements in Core Numeric Generation**:
  - Implemented a factory pattern for function compilation with unified `compile_function` interface
  - Created standardized error handling with `handle_error` function for consistent messaging
  - Simplified mesh grid creation using NumPy's capabilities more directly
  - Unified shock grid generation through `build_shock_grid` factory function
  - Optimized parameter resolution with loop prevention and improved type handling
  - Eliminated code duplication by consolidating common functionality
  - Enhanced parameter handling with common helper functions
  - Improved function dependency resolution with clearer organization

### Fixed
- **Enhanced Numeric Functionality**:
  - Fixed Markov process shock grid generation to handle additional parameters
  - Improved shock grid generation with proper parameter inheritance
  - Ensured correct handling of multi-output functions and scalar accessors
  - Fixed parameter resolution for string references with maximum recursion depth
  - Enhanced type checking and conversions for more robust numeric operations

### Documentation
- **Better Interfaces and Method Documentation**:
  - Added comprehensive docstrings for all refactored functions
  - Clarified parameter descriptions and return value documentation
  - Documented factory pattern usage and implementation details
  - Improved type hints for better IDE support and code completion

## [0.1.5.16.dev0] - 2025-04-18

### Added
- **Multi-Dimensional Function Support (R^N → R^M)**:
  - Implemented native support for functions returning multiple outputs
  - Added named output capabilities with OrderedDict return values
  - Extended YAML configuration for multi-output function definitions:
    - New `outputs` array for declaring output variable names
    - New `exprs` mapping for defining multiple expressions by output name
    - Support for vector axis specification and broadcasting behavior
  - Created auto-generated scalar sub-functions for each output component (e.g., `function.output_name`)
  - Added comprehensive function signature discovery for improved auto-documentation
  - Implemented runtime shape validation for vectorized operations
  
### Changed
- **Function Architecture Improvements**:
  - Enhanced `generate_numerical_functions` to handle multi-output functions
  - Updated compilation logic to create structured result dictionaries
  - Improved parameter handling with consistent vectorized broadcasting
  - Updated movers to properly consume multi-output function results
  - Enhanced docstring generation for function callables
  
### Examples
- Added `examples/load_consind_multi.py` demonstrating multi-output functions
- Updated `examples/cons_indshock/ConsInd_multi.yml` with multi-output function examples:
  - `util_and_mutil`: Joint calculation of utility and marginal utility
  - `transition_budget`: Combined asset transition and savings rate calculation
  - `asset_transition_value`: Combined state transition and value calculation
  - `egm_operations`: Comprehensive EGM implementation in a single function

## [0.1.5.15.dev0] - 2025-04-20

### Added
- **DiscreteMarkov Shock Processes**:
  - New `DiscreteMarkov` shock type for automatic discretization of AR(1) processes
  - Pure NumPy implementation without external dependencies
  - Support for both Tauchen and Rouwenhorst discretization methods
  - Automatically generates both shock values and transition matrices
  - Accessed through index-based state variables for efficiency
  
- **Multi-dimensional State Spaces**:
  - Support for compound state spaces combining shock indices with continuous variables
  - Automated mesh grid generation for multi-dimensional state spaces
  - Flexible configuration for different grid types (uniform, log, etc.) per dimension
  - Comprehensive access methods for working with grid values and indices
  - Efficient lookup between state indices and economic values

- **YAML Configuration Enhancements**:
  - Extended schema for defining Markov processes and grid configurations
  - Simple reference syntax for global parameters and settings
  - Clear separation between shock definition and state space configuration

### Improved
- **Cleaner Modeling Approach**:
  - Separation of computational indices from economic shock values
  - Efficient state representation using integer indices for discrete components
  - Automatic mapping between indices and actual values
  - Flexible grid generation for different state variables

### Examples
- Added `examples/markov_shock_example.py` demonstrating the DiscreteMarkov shock grid
- Added `examples/multidim_example.py` showing multi-dimensional state spaces

## [0.1.5.14.dev0] - 2025-04-15

### Added
- **Multi-target Stages Support**:
  - Fan-out capability for backward direction: `arvl` → many `cntn`
  - Fan-in capability for forward direction: many `cntn` → `arvl`
  - Perch `.sol` and `.dist` can now be dictionaries for fan-in container functionality
  - New attributes in Mover class: `branch_key` and `agg_rule`
  
- **Unified Transpose Builder Mechanism**:
  - Single canonical mechanism for auto-generating transpose movers
  - Clean implementation across Stage, Period, and ModelCircuit layers
  - Non-recursive design for better maintainability
  - Branch-key awareness for proper mapping in complex models
  
- **Branch-key Propagation and Collision Detection**:
  - Branch-key propagation with clear collision warnings
  - Auto-generation of unique keys when duplicates detected
  - Improved handling of multiple incoming edges
  - Clear warning messages for potential branch-key collisions

### Changed
- **Extended Period API**:
  - New `connect_bwd` and `connect_fwd` methods with branch_key support
  - Improved solver execution flow for fan-in and fan-out patterns
  - Comprehensive dict-based container model for multi-source values
  - Default branch-key auto-generation from source stage ID

### Fixed
- Fixed potential cycles in stage graphs by making transpose creation optional
- Improved collision detection for duplicate connections

### Examples
- Added comprehensive demo in `examples/multi_target_demo.py`

## [0.1.5.13.dev0] - 2025-04-13

### Changed
- **Renamed Package from CircuitCraft to StageCraft**:
  - Renamed core package from `circuitcraft` to `stagecraft` to better reflect its focus on stage-based modeling
  - Updated all import statements throughout the codebase
  - Updated documentation, examples, and test files to reference the new package name
  - Added note to README about the package renaming
  - Marked the task of renaming the package as completed in Todo.md
  - Updated directory structure diagram in README.md

### Added
- **Directory Structure Documentation**:
  - Added comprehensive directory structure documentation to README.md
  - Created a visual directory tree showing the organization of the codebase
  - Added detailed descriptions of key directories and their purposes
  - Included explanations of different module categories

### Improved
- **Backward Compatibility**:
  - Removed compatibility layer originally planned to maintain backward compatibility with `circuitcraft`
  - Made a clean break to the new package name for better long-term maintainability

### Fixed
- **Import Path Handling**:
  - Fixed import paths in example files and test scripts
  - Updated module references in DEGM whisperer implementation
  - Fixed import fallbacks in ConsInd.py and other example files
  - Enhanced error handling for imports with more descriptive error messages

### Updated
- **Todo.md Formatting**:
  - Added proper GitHub-compatible checkboxes to Todo.md
  - Applied consistent formatting throughout the Todo file
  - Added emoji icons to section headings for visual distinction
  - Added numerical section headings for better organization
  - Added new section for CDC-Bellman comparison and merging

## [0.1.5.12.dev0] - 2025-04-09

- Model sequence is now model circuit
- Re-organized developer's prompts and todos into a dev directory


## [0.1.5.11.dev0] - 2025-04-09

- Model horse in model directory
- Reorganize heptapod plugin to be a /src module
- Add model overview notebook

## [0.1.5.10.dev0] - 2025-04-09

### Added
- **Dual Solver Approach with OperatorFactory and Whisperer**:
  - Added `model_mode` parameter to `Stage` class to support both `"in_situ"` and `"external"` modes
  - Created `operator_factory` attribute for in-situ mode and maintained `whisperer` for external mode
  - Implemented `attach_operatorfactory_operators()` method for in-situ operator attachment
  - Maintained backward compatibility with `attach_whisperer_operators()` method
  - Added example external solver implementation with `whisperer_external` function
  - Updated `lifecycle_t1_stage.py` to demonstrate both approaches with comparison
  - Generated visualization outputs for both modes to allow direct comparison

### Improved
- **Code Clarity and Architecture**:
  - Clearer separation between in-situ operator factories and external whisperer solvers
  - Enhanced flexibility with support for both internal and external solution approaches
  - Better alignment with dynamic programming and external solver integration patterns
  - Simplified integration with specialized numeric libraries through external mode
  - Improved documentation describing both approaches and their use cases

### Fixed
- **Enhanced Solver Integration**:
  - Resolved inconsistencies in operator attachment and solver signaling
  - Improved error handling when no solver is provided
  - Added proper warning messages for deprecated method usage
  - Ensured both approaches produce identical numeric results for verification

All notable changes to CircuitCraft will be documented in this file.

## [0.1.5.09.dev0] - 2025-04-09

### Changed
- **Standardized Mover Naming Convention**:
  - Refactored `Mover`, `Stage`, `Period`, `ModelSequence`, and `CircuitBoard` to consistently use `source_name`, `target_name`, `source_keys`, and `target_key`.
  - Ensured `source_name` and `target_name` always reflect the actual data source and destination nodes for the specific mover direction (forward or backward), eliminating ambiguity.
  - Updated `ModelSequence` movers (`add_inter_period_connection`, `_create_transpose_mover`) to use global node IDs (e.g., `p0:stageA`) for `source_name` and `target_name`.
- **Removed Redundant Mover Attributes**:
  - Eliminated the setting and reliance on `mover.source_perch_attr` and `mover.target_perch_attr` in `Period` and `ModelSequence`, as this information is now correctly captured by `mover.source_keys` and `mover.target_key`.
- **Refactored Transpose Logic**:
  - Updated `ModelSequence._create_transpose_mover` to robustly derive necessary perch information by parsing the original mover's `source_keys` and `target_key`, instead of relying on the removed redundant attributes.
  - Confirmed `CircuitBoard.create_transpose_connections` correctly swaps `source_name` and `target_name`.

### Improved
- **Code Consistency and Maintainability**: Unified the way movers are defined and referenced across different levels of the framework (Stage, Period, ModelSequence).
- **Reduced Confusion**: Eliminated potentially conflicting or redundant attributes on Mover objects.

### Fixed
- Ensured correct identification of source/target nodes and keys when creating transpose movers, especially in `ModelSequence`.

### Testing
- Verified changes by successfully running example scripts: `monetary_policy_mortgage_model.py`, `worker_retiree_demo.py`, and `examples/heptapod/lifecycle_t1_stage.py`.

## [0.1.5.08.dev0] - 2025-08-01

### Changed
- **Standardized Mover Naming Convention**:
  - Refactored `Mover`, `Stage`, `Period`, `ModelSequence`, and `CircuitBoard` to consistently use `source_name`, `target_name`, `source_keys`, and `target_key`.
  - Ensured `source_name` and `target_name` always reflect the actual data source and destination nodes for the specific mover direction (forward or backward), eliminating ambiguity.
  - Updated `ModelSequence` movers (`add_inter_period_connection`, `_create_transpose_mover`) to use global node IDs (e.g., `p0:stageA`) for `source_name` and `target_name`.
- **Removed Redundant Mover Attributes**:
  - Eliminated the setting and reliance on `mover.source_perch_attr` and `mover.target_perch_attr` in `Period` and `ModelSequence`, as this information is now correctly captured by `mover.source_keys` and `mover.target_key`.
- **Refactored Transpose Logic**:
  - Updated `ModelSequence._create_transpose_mover` to robustly derive necessary perch information by parsing the original mover's `source_keys` and `target_key`, instead of relying on the removed redundant attributes.
  - Confirmed `CircuitBoard.create_transpose_connections` correctly swaps `source_name` and `target_name`.

### Improved
- **Code Consistency and Maintainability**: Unified the way movers are defined and referenced across different levels of the framework (Stage, Period, ModelSequence).
- **Reduced Confusion**: Eliminated potentially conflicting or redundant attributes on Mover objects.

### Fixed
- Ensured correct identification of source/target nodes and keys when creating transpose movers, especially in `ModelSequence`.

### Testing
- Verified changes by successfully running example scripts: `monetary_policy_mortgage_model.py`, `worker_retiree_demo.py`, and `examples/heptapod/lifecycle_t1_stage.py`.

## [0.1.5.07.dev0] - 2025-04-09

### Fixed
- **Robust Transpose Connection Handling for Discrete Stages**:
  - Fixed critical issue with backward connections to discrete stages requiring branch_key parameters
  - Modified `period.py` to automatically generate branch keys when creating transpose connections to discrete stages
  - Implemented intelligent branch_key detection that defaults to target stage ID if none is provided
  - Added warning messages when auto-generating branch keys to inform users of the automatic behavior
  - Ensures proper data flow in backward solving when using the create_all_transpose_connections method

### Improved
- **Enhanced Model Construction Resilience**:
  - Made intra-period and inter-period connections more robust when involving discrete choice stages
  - Resolved connection failures in worker_retiree_demo.py and monetary_policy_mortgage_model.py examples
  - Eliminated need for manual branch_key assignments in most simple connection patterns
  - Preserved backward compatibility with existing models that already specify branch keys

### Documentation
- **Better Error Messaging**:
  - Added clear warning messages that explain when and why branch keys are being auto-generated
  - Made error messages more descriptive when branch keys are required but missing
  - Provided helpful context when generating transpose connections automatically

## [0.1.5.06.dev0] - 2025-04-09

### Changed
- **Mover Class Architectural Simplification**:
  - Removed deprecated parameters (`map_data`, `parameters`, `numerical_hyperparameters`)
  - Extracted complex conversion logic to standalone `convert_legacy_model()` utility function
  - Simplified model initialization with cleaner parameter structure
  - Improved type hints to accept `Any` for model types rather than only dictionaries
  - Enhanced documentation with clearer parameter explanations
  - Added `legacy_conversion` parameter to control automatic format conversion

### Added
- **Enhanced Image Organization**:
  - Implemented consistent pattern for image file storage with model-specific subdirectories
  - Created dedicated subdirectories (`/images/mortgages/`, `/images/worker_retiree/`, `/images/heptapod/`)
  - Added automatic directory creation checks in visualization scripts
  - Updated all visualization functions to accept path parameters

- **Visualization Flexibility**:
  - Added configurable node color mappings for different stage types
  - Implemented customizable edge style mappings for various edge types
  - Added support for multiple label formats (full names and abbreviated names)
  - Added capability to mark special nodes (initial and terminal)
  - Implemented new "period spring" layout option to group nodes by period

### Improved
- **Code Organization**:
  - Consistent parameter handling across visualization functions
  - Enhanced cross-platform compatibility using `os.path.join` for all file paths
  - Better organization of operator factory functions with clear documentation
  - Improved error handling and validation throughout the codebase
  - Clearer separation between model representations and computational implementations

## [0.1.5.05.dev0] - 2025-04-07

### Added
- **Jupyter Notebook for Stage Class Workflow**:
  - Created interactive notebook demonstrating the complete lifecycle Stage model workflow
  - Implemented clean, focused examples showing each step of the Stage initialization, configuration, and solution process
  - Documented both internal (whisperer-based) and external solution approaches
  - Added visualizations for consumption policy functions and value functions
  - Demonstrated time iteration for multi-period lifecycle models
  - Provided clear illustrations of the CDC (Continuation, Decision, Arrival) pattern and data flow
  - Enhanced accessibility for new users to understand the Stage class architecture

### Documentation
- **Improved Model Architecture Documentation**:
  - Added visualization of the CDC pattern and component relationships
  - Documented the workflow for transferring value functions across time periods
  - Clarified the separation between mathematical models and computational implementations
  - Explained the Endogenous Grid Method (EGM) implementation in accessible terms
  - Structured demonstrations around key architectural principles of ModCraft

## [0.1.5.04.dev0] - 2025-04-09

### Changed
- **Data Flow Enhancement in Circuit Board Class**:
  - Modified how dictionary results are handled in the `execute_mover` method
  - When a mover returns a dictionary and has a target_key specified, the entire dictionary is now stored directly under that target_key
  - Previous behavior only worked if dictionary keys matched perch data keys exactly
  - This ensures consistent data flow between operators and perches, especially for backward solvers in dynamic programming

### Fixed
- **Improved Stage and Circuit Board Interaction**:
  - Ensured that mover result dictionaries are properly stored in perch attributes regardless of their structure
  - Fixed issues where backward-solving dynamic programs would fail to properly update perch data
  - Addressed a bug where directly calling operators vs using the solver framework could produce different results
  
### Improved
- **Reduced Debug Output**:
  - Removed verbose debug prints showing detailed mover results
  - Commented out value update notifications in the solve_backward method
  - Maintained critical error messages and warnings for failing movers
  - Modified Stage's `attach_whisperer_operators` to only check backward movers for portability
  - Updated Stage to use more concise status messages

## [0.1.5.03.dev0] - 2025-04-07

**HIGHLY PRELIMINARY AND UNTESTED**

### Changed
- **Updated Mover Object References in Period Class**:
  - Restructured `add_connection` and `add_inter_period_connection` methods to use attribute strings/keys
  - Removed direct object references from mover objects
  - Movers now reference perches through attribute strings, making the code more maintainable
  - Updated documentation to reflect the new reference approach

### Improved
- **Enhanced Period and ModelSequence Stability**:
  - Modified `solve_backward` and `solve_forward` methods to work with the attribute-based approach
  - Added error handling for missing perches in both methods
  - Improved debugging information during execution
  - Standardized the way nodes and edges are created and accessed

### Fixed
- **Visualization Improvements**:
  - Period numbers are now properly displayed in node labels for better clarity
  - Standardized label creation across different visualization methods
  - Fixed potential issues with missing object references during graph traversal

### Documentation
- **Updated Implementation Descriptions**:
  - Added explanatory comments to the refactored code sections
  - Clarified the data flow between perches and movers
  - Enhanced documentation of all modified methods

## [0.1.5.02.dev0] - 2025-04-07

**HIGHLY PRELIMINARY AND UNTESTED**

### Added
- **Enhanced Stage Graph Visualization**:
  - Added period numbers to node labels in `visualize_stage_graph` method to improve clarity
  - Implemented custom labels for specific node types (worker, retiree, discrete choice)
  - Added a new `include_period_in_label` parameter to control period number display in labels
  - Created `examples/worker_retiree_demo.py` to demonstrate the custom labeling feature

### Improved
- **Model Sequence Visualization**:
  - Enhanced node label readability with period prefixes (e.g., "P0: Worker", "P2: Discrete Choice")
  - Improved color coding for different stage types
  - Made the visualization legend more informative with custom node type labels
  - Fixed connections between periods in the worker_retiree_demo to properly model forward flow

### Documentation
- **Enhanced Visualization Examples**:
  - Added detailed example for worker/retiree lifecycle model visualization
  - Demonstrated both intra-period and inter-period connection patterns
  - Included examples for different visualization layouts (forward-only, all edges, spring layout)
  - Generated PNG visualizations for reference and demonstration

## [0.1.5.01.dev0] - 2025-04-06

**HIGHLY PRELIMINARY AND UNTESTED**

### Added
- **Enhanced Eulerian Sub-Graphs Visualization for Pension Models**:
  - Added visualization scripts that accurately represent the computational structure of pension models using Eulerian sub-graphs
  - Created `

## [0.1.8.dev7] - YYYY-MM-DD

### Added
- **Config Override in `load_circuit`**:
  - Introduced `cfg_override` parameter to `dynx.stagecraft.io.load_circuit`.
  - Allows callers to provide an in-memory configuration dictionary, bypassing the loading of YAML files from the saved bundle.
  - This is useful for re-running models with modified parameters without altering the saved bundle, or for loading bundles where on-disk configs might be stale or malformed.
  - Added a validation step to ensure `cfg_override` contains the required keys: "master", "stages", and "connections".

### Changed
- **`CircuitRunner` Integration**:
  - `dynx.runner.circuit_runner.CircuitRunner` now utilizes the `cfg_override` feature.
  - When `load_if_exists` is true (or `__runner.mode` is "load"), `CircuitRunner` first patches the `base_cfg` with the current parameter vector (`x`) to create a `cfg_fresh`.
  - This `cfg_fresh` is then passed as `cfg_override` to `load_circuit`.
  - This ensures that loaded models are always initialized with parameters consistent with the current sweep iteration, preventing issues with stale on-disk configurations (e.g., "float() argument must be a string or a real number, not 'list'" error due to incorrect linspace parsing from an old config).
- **Type Hint Update**:
  - Updated the type hint for `cfg_override` in `CircuitRunner._maybe_load_bundle` to `Optional[Dict[str, Any]]`.

### Fixed
- **Stale Configuration Issue**:
  - Addresses a bug where loading a previously saved model bundle could use outdated configuration files from disk, leading to errors if the configuration schema or parameter types had changed (e.g., linspace parameters being loaded as lists instead of being evaluated).
  - By using `cfg_override`, the model is now always initialized with a configuration that is consistent with the current `CircuitRunner`'s `base_cfg` and the parameter vector `x`.

### Documentation
- **Unit Test**:
  - Added `test_cfg_override_load` to `tests/unit/test_runner_deviation_comprehensive.py` to verify the config override mechanism.

### Migration Notes
- This change is backward compatible. Existing code that calls `load_circuit` without `cfg_override` will continue to load configurations from disk as before.
- If using `CircuitRunner` with `load_if_exists=True`, models loaded from disk will now be fresher and less prone to errors caused by stale configurations.