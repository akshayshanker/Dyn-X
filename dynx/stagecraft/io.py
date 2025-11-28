"""
dynx.stagecraft.io
------------------
I/O utilities for saving and loading ModelCircuit artifacts.

This module provides:
- save_circuit: Save a ModelCircuit and its artifacts to a directory
- load_circuit: Load a ModelCircuit from a saved directory
- load_config: Load configuration files from a structured directory
"""

from __future__ import annotations

import argparse
import datetime as _dt
import pickle
import logging
import shutil
from pathlib import Path
from typing import Any, List, Dict, Union, Optional
import gc

import yaml

from dynx.heptapodx.io.yaml_loader import load_config as load_yaml_file
from dynx.stagecraft import makemod
from dynx.stagecraft.solmaker import Solution

logger = logging.getLogger("dynx.stagecraft.io")

__all__ = [
    "save_circuit",
    "load_circuit",
    "load_config",
    "stamp_model_id",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _safe_pickle(obj: Any, path: Path) -> None:
    """Pickle *obj* with protocol 5."""
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=5)


def _dump_object(obj: Any, path: Path) -> None:
    """
    Best-effort pickle helper.

    Order of preference
    -------------------
    1. obj.pkl(path)             # e.g. dynx.stagecraft.solmaker.Solution
    2. Dict with Solution values # e.g. {"from_owner": <Solution>, "from_renter": <Solution>}
    3. pickle obj.as_dict()      # if obj exposes .as_dict()
    4. plain pickle              # protocol 5
    """
    try:
        if hasattr(obj, "pkl") and callable(obj.pkl):
            obj.pkl(path)                      # type: ignore[attr-defined]
            return

        # Handle dict whose values are Solution objects (e.g., TENU branches)
        if isinstance(obj, dict) and any(isinstance(v, Solution) for v in obj.values()):
            serialized = {
                k: (v.as_dict() if isinstance(v, Solution) else v)
                for k, v in obj.items()
            }
            _safe_pickle(serialized, path)
            return

        if hasattr(obj, "as_dict") and callable(obj.as_dict):
            _safe_pickle(obj.as_dict(), path)
            return

        _safe_pickle(obj, path)

    except (pickle.PicklingError, TypeError) as exc:
        logger.warning("Skipping %s – not pickle-able (%s)", path.name, exc)

def _dump_yaml(data: dict, path: Path) -> None:
    """Helper: write *data* to *path* (create parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
# -----------------------------------------------------


def _copy_configs(src: str | Path | dict | List[Any], dst: Path) -> None:
    """Copy directories / files *or* dump Python-dict configs to YAML."""

    dst.mkdir(parents=True, exist_ok=True)

    # normalise to an iterable
    if not isinstance(src, (list, tuple)):
        src = [src]

    for item in src:
        # ------------------------------------------------------------------
        # 0) Canonical container dict  {"master": …, "stages": …, "connections": …}
        # ------------------------------------------------------------------
        if (
            isinstance(item, dict)
            and {"master", "stages", "connections"} <= set(item.keys())
        ):
            _dump_yaml(item["master"], dst / "master.yml")
            for stage_name, stage_cfg in item["stages"].items():
                _dump_yaml(stage_cfg, dst / "stages" / f"{stage_name}.yml")
            _dump_yaml(item["connections"], dst / "connections.yml")
            continue  # done with this item

        # ------------------------------------------------------------------
        # 1) Single *flat* config dict  (previous behaviour)
        # ------------------------------------------------------------------
        if isinstance(item, dict):
            if item.get("model_type", "").lower() == "master" or "horizon" in item:
                _dump_yaml(item, dst / "master.yml")
            elif item.get("connections") is True or "edges" in item:
                _dump_yaml(item, dst / "connections.yml")
            else:  # assume stage config
                stage_name = item.get("name", "stage").upper()
                _dump_yaml(item, dst / "stages" / f"{stage_name}.yml")
            continue

        # ------------------------------------------------------------------
        # 2)  Pathlike sources (dir or single file)
        # ------------------------------------------------------------------
        p = Path(item).expanduser().resolve()
        if p.is_dir():
            for fn in p.rglob("*"):
                if fn.suffix.lower() in {".yml", ".yaml", ".json", ".toml"}:
                    rel = fn.relative_to(p)
                    (dst / rel.parent).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(fn, dst / rel)
        elif p.is_file():
            shutil.copy2(p, dst / p.name)
        else:
            logger.warning("Config source %s not recognised; skipped", item)


def stamp_model_id(circuit: Any, version: str | None = None) -> str:
    root = getattr(circuit, "name", "model")
    today = _dt.date.today().isoformat()
    if version is None:
        version = getattr(circuit, "version", "dev")
    return f"{root}_{version}_{today}"


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(folder_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration files from a structured directory.
    
    Expected structure:
    ```
    folder_path/
    ├── master.yml
    ├── stages/
    │   ├── stage1.yml
    │   └── stage2.yml
    └── connections.yml
    ```
    
    Stage names are automatically uppercased to match Stage object naming conventions.
    For example, 'tenu.yml' will be loaded with key 'TENU' in the stages dictionary.
    Both .yml and .yaml file extensions are supported.
    
    Parameters
    ----------
    folder_path : str or Path
        Path to the configuration directory
        
    Returns
    -------
    dict
        Dictionary with keys "master", "stages", and "connections".
        Stage names in the "stages" dictionary are uppercased.
        
    Raises
    ------
    FileNotFoundError
        If required files are missing
    """
    folder = Path(folder_path).expanduser().resolve()
    
    if not folder.exists():
        raise FileNotFoundError(f"Configuration directory not found: {str(folder)}")
    
    # Load master configuration
    master_file = folder / "master.yml"
    if not master_file.exists():
        raise FileNotFoundError(f"master.yml not found in {str(folder)}")
    master_config = load_yaml_file(master_file)
    
    # Load connections configuration
    connections_file = folder / "connections.yml"
    if not connections_file.exists():
        raise FileNotFoundError(f"connections.yml not found in {str(folder)}")
    connections_config = load_yaml_file(connections_file)
    
    # Load stage configurations from stages/ subdirectory
    stages_dir = folder / "stages"
    if not stages_dir.exists():
        raise FileNotFoundError(f"stages/ directory not found in {str(folder)}")
    
    stage_configs = {}
    # Support both .yml and .yaml extensions
    patterns = ("*.yml", "*.yaml")
    yml_paths = [p for pattern in patterns for p in stages_dir.glob(pattern)]
    
    for yml_file in yml_paths:
        stage_name = yml_file.stem.upper()  # Upper-case to match Stage names
        stage_configs[stage_name] = load_yaml_file(yml_file)
    
    if not stage_configs:
        raise FileNotFoundError(f"No stage configuration files found in {str(stages_dir)}")
    
    return {
        "master": master_config,
        "stages": stage_configs,
        "connections": connections_config
    }


# ---------------------------------------------------------------------------
# main API
# ---------------------------------------------------------------------------

def save_circuit(
    circuit: Any,
    dest: str | Path,
    config_src: str | Path | List[Path],
    model_id: str | None = None,
) -> Path:
    """
    Save configs and every *.sol / .sim* attached to a ModelCircuit.

    Returns the path to ``<dest>/<model_id>``.
    """
    dest = Path(dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    if model_id is None:
        model_id = stamp_model_id(circuit)

    target_dir = dest / model_id
    if target_dir.exists():
        logger.warning("Overwriting existing directory: %s", target_dir)
        # More robust directory removal
        import time
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                shutil.rmtree(target_dir)
                break
            except OSError as e:
                if attempt < max_attempts - 1:
                    logger.warning(f"Failed to remove directory (attempt {attempt+1}): {e}")
                    time.sleep(0.5)  # Wait a bit
                    # Try to force cleanup
                    gc.collect()
                else:
                    # Last attempt - try to rename instead
                    backup_name = f"{target_dir}_{int(time.time())}_backup"
                    logger.warning(f"Cannot remove directory, renaming to: {backup_name}")
                    target_dir.rename(backup_name)
    target_dir.mkdir()

    # 1) copy configs verbatim
    _copy_configs(config_src, target_dir / "configs")

    # 2) discover periods robustly
    if hasattr(circuit, "periods_list"):
        periods_iter = list(circuit.periods_list)
    elif hasattr(circuit, "periods"):
        pa = circuit.periods
        periods_iter = list(pa.values() if isinstance(pa, dict) else pa)
    else:
        raise AttributeError("Circuit has no 'periods_list' or 'periods' attribute")

    if not periods_iter:
        raise ValueError("Circuit contains zero periods; nothing to save")

    data_dir = target_dir / "data"
    files_written: list[str] = []

    # 3) walk periods → stages → perches
    for p_idx, period in enumerate(periods_iter):
        period_dir = data_dir / f"period_{p_idx}"

        stages_attr = getattr(period, "stages", {})
        stages_iter = stages_attr.values() if isinstance(stages_attr, dict) else stages_attr

        for stage in stages_iter:
            stage_dir = period_dir / stage.name

            for perch_name, perch in getattr(stage, "perches", {}).items():
                perch_dir = stage_dir / perch_name
                perch_dir.mkdir(parents=True, exist_ok=True)

                # ---- sol ---------------------------------------------------
                if getattr(perch, "sol", None) is not None:
                    sol_path = perch_dir / "sol.pkl"
                    _dump_object(perch.sol, sol_path)
                    files_written.append(sol_path.relative_to(target_dir).as_posix())

                # ---- sim / dist -------------------------------------------
                sim_obj = (
                    getattr(perch, "sim", None)
                    if hasattr(perch, "sim")
                    else getattr(perch, "dist", None)
                )
                if sim_obj is not None:
                    sim_path = perch_dir / "sim.pkl"
                    _dump_object(sim_obj, sim_path)
                    files_written.append(sim_path.relative_to(target_dir).as_posix())

    # 4) write manifest
    manifest = {
        "model_id": model_id,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "files": sorted(files_written),
    }
    (target_dir / "manifest.yml").write_text(yaml.safe_dump(manifest, sort_keys=False))

    return target_dir


def load_circuit(
    saved_dir: Union[str, Path],
    *,
    restore_data: bool = True,
    cfg_override: Optional[dict] = None,
    periods_to_load: Optional[List[int]] = None,
    stages_to_load: Optional[Dict[int, List[str]]] = None,
) -> Any:
    """
    Load a ModelCircuit from a saved directory.
    
    Parameters
    ----------
    saved_dir : str or Path
        Path to the directory created by save_circuit
    restore_data : bool, default True
        Whether to restore solution/simulation data to perches
    cfg_override : dict, optional
        Override configuration dictionary
    periods_to_load : list of int, optional
        List of period indices to load. If None, loads all periods.
        Example: [0, 1] loads only periods 0 and 1
    stages_to_load : dict of int to list of str, optional
        Dictionary mapping period indices to lists of stage names to load.
        If None for a period, loads all stages in that period.
        Example: {0: ["OWNC"], 1: None} loads only OWNC in period 0, all stages in period 1
        
    Returns
    -------
    ModelCircuit
        The reconstructed ModelCircuit with attached solutions
        
    Raises
    ------
    FileNotFoundError
        If the saved directory or required files are missing
    """
    saved_dir = Path(saved_dir).expanduser().resolve()
    
    if not saved_dir.exists():
        raise FileNotFoundError(f"Saved directory not found: {str(saved_dir)}")
    
    # Validate manifest exists
    manifest_path = saved_dir / "manifest.yml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.yml not found in {str(saved_dir)}")
    
    manifest = yaml.safe_load(manifest_path.read_text())
    logger.info(f"Loading ModelCircuit '{manifest.get('model_id', 'unknown')}'")
    
    # ---------------------------------------------------------------------
    # 1) decide where configs come from
    # ---------------------------------------------------------------------
    if cfg_override is not None:
        if not {"master", "stages", "connections"} <= set(cfg_override.keys()):
            raise ValueError("cfg_override must contain 'master', 'stages', 'connections'")
        cfg = cfg_override  # caller provided a fresh container
    else:
        configs_dir = saved_dir / "configs"
        if not configs_dir.exists():
            raise FileNotFoundError(f"configs directory not found in {str(saved_dir)}")
        cfg = load_config(configs_dir)  # ORIGINAL behaviour

    # Initialize the model circuit
    circuit = makemod.initialize_model_Circuit(
        master_config=cfg["master"],
        stage_configs=cfg["stages"],
        connections_config=cfg["connections"]
    )
    makemod.compile_all_stages(circuit)
    
    
    # Restore data if requested
    if not restore_data:
        return circuit
    
    data_dir = saved_dir / "data"
    if not data_dir.exists():
        logger.info("No 'data' directory found – returning circuit without attachments")
        return circuit

    # Walk over saved folders; map onto circuit structure
    for period_dir in data_dir.glob("period_*"):
        try:
            p_idx = int(period_dir.name.split("_")[1])
        except (IndexError, ValueError):
            logger.warning("Skipping unexpected folder: %s", period_dir)
            continue
            
        # Skip periods not in periods_to_load if specified
        if periods_to_load is not None and p_idx not in periods_to_load:
            logger.debug(f"Skipping period {p_idx} (not in periods_to_load)")
            continue
            
        try:
            period = circuit.periods_list[p_idx]
        except IndexError:
            logger.warning("Saved data has extra period %s", p_idx)
            continue

        for stage_dir in period_dir.iterdir():
            if not stage_dir.is_dir():
                continue
            stage_name = stage_dir.name
            
            # Skip stages not in stages_to_load for this period if specified
            if stages_to_load is not None and p_idx in stages_to_load:
                if stages_to_load[p_idx] is not None and stage_name not in stages_to_load[p_idx]:
                    logger.debug(f"Skipping stage {stage_name} in period {p_idx} (not in stages_to_load)")
                    continue
            
            stage = next((s for s in period.stages.values() if s.name == stage_name), None)
            if stage is None:
                logger.warning("Stage %s not found in circuit", stage_name)
                continue

            for perch_dir in stage_dir.iterdir():
                if not perch_dir.is_dir():
                    continue
                perch_name = perch_dir.name
                perch = stage.perches.get(perch_name)
                if perch is None:
                    logger.warning("Perch %s missing in stage %s", perch_name, stage_name)
                    continue

                sol_path = perch_dir / "sol.pkl"
                if sol_path.exists():
                    try:
                        with sol_path.open("rb") as f:
                            obj = pickle.load(f)
                        
                        # If the pickled object is a plain dict that looks
                        # like a Solution dump, convert it back.
                        if isinstance(obj, dict) and {"policy", "EGM", "timing"} <= obj.keys():
                            obj = Solution.from_dict(obj)
                        
                        # Handle branch dictionaries (e.g., TENU with "from_owner", "from_renter")
                        elif isinstance(obj, dict):
                            # Re-hydrate each branch if it looks like a Solution dict
                            for key, val in obj.items():
                                if isinstance(val, dict) and {"policy", "EGM", "timing"} <= val.keys():
                                    obj[key] = Solution.from_dict(val)
                        
                        perch.sol = obj
                        del obj
                        gc.collect()
                    except Exception as exc:
                        logger.exception("Failed to load sol: %s", sol_path)

                sim_path = perch_dir / "sim.pkl"
                if sim_path.exists():
                    try:
                        with sim_path.open("rb") as f:
                            obj = pickle.load(f)
                        
                        # If the pickled object is a plain dict that looks
                        # like a Solution dump, convert it back.
                        if isinstance(obj, dict) and {"policy", "EGM", "timing"} <= obj.keys():
                            obj = Solution.from_dict(obj)
                        
                        # Handle branch dictionaries (e.g., TENU with "from_owner", "from_renter")
                        elif isinstance(obj, dict):
                            # Re-hydrate each branch if it looks like a Solution dict
                            for key, val in obj.items():
                                if isinstance(val, dict) and {"policy", "EGM", "timing"} <= val.keys():
                                    obj[key] = Solution.from_dict(val)
                        
                        perch.sim = obj
                        del obj
                        gc.collect()
                    except Exception as exc:
                        logger.exception("Failed to load sim: %s", sim_path)

    logger.info("Loaded ModelCircuit with saved solutions from %s", saved_dir)
    return circuit


def _cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="dynx-saver", description="Save / load DynX ModelCircuit artifacts")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # save -----------------------------------------------------------------
    p_save = sub.add_parser("save", help="Save a ModelCircuit")
    p_save.add_argument("config", type=str, help="Path to config directory (or file)")
    p_save.add_argument("dest", type=str, help="Destination directory to store bundle")
    p_save.add_argument("--model-id", type=str, default=None, help="Explicit model identifier")
    p_save.add_argument("--module", type=str, default="dynx.build", help="Module that builds the circuit (expects build() function)")

    # load -----------------------------------------------------------------
    p_load = sub.add_parser("load", help="Load a previously saved ModelCircuit")
    p_load.add_argument("saved_dir", type=str, help="Path to saved bundle directory")

    args = parser.parse_args(argv)

    if args.cmd == "save":
        # Dynamic import build module (user may provide custom script building a circuit)
        build_mod = __import__(args.module, fromlist=["build"])
        if not hasattr(build_mod, "build"):
            parser.error(f"Module {args.module} has no 'build()' function")
        circuit = build_mod.build(args.config)
        save_circuit(circuit, args.dest, args.config, model_id=args.model_id)
        makemod.compile_all_stages(circuit)
    elif args.cmd == "load":
        load_circuit(args.saved_dir)
    else:
        parser.error("Unknown command")


