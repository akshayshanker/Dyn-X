"""
Deviation metrics for comparing model policies against reference solutions.

This module provides metric functions that compute L2 and L∞ norms of the
difference between a model's policy and a reference model's policy.
"""

from typing import Any, Callable, Literal, Optional

import numpy as np

from dynx.runner.circuit_runner import CircuitRunner
from dynx.runner.reference_utils import load_reference_model
from dynx.runner.metric_requirements import get_metric_requirements


import numpy as np
from typing import Any, Optional

def _extract_policy(
    model: Any,
    stage: str = "OWNC",
    sol_attr: str = "policy",
    key: str = "c",
    period_idx: int | str | None = 0,  # -1 ⇒ last period; "first" ⇒ 0
) -> Optional[np.ndarray]:
    """
    Extract a policy array from a ModelCircuit.

    Parameters
    ----------
    model : ModelCircuit
    stage : str
        Stage name to extract from (default: "OWNC").
    sol_attr : str
        Attribute of the Solution object to look for (default: "policy").
    key : str
        Key/attribute inside the policy object to extract (default: "c").
        Use '' or None when the policy is itself an ndarray.
    period_idx : int | str | None
        • int  -> explicit period index  
        • -1   -> last period (default)  
        • "first" -> period 0  
        • None -> search all periods until the stage is found
    """
    # ---------------------------------------------------------------------
    # 1) Locate the right Period
    # ---------------------------------------------------------------------
    period_obj = None

    if not hasattr(model, "periods_list"):
        # Not a ModelCircuit (maybe a single-period model) – give up early
        return None

    periods = model.periods_list
    if not periods:
        return None

    if period_idx is None:
        # Search every period until we find the stage
        for p in periods:
            if hasattr(p, "get_stage"):
                try:
                    if p.get_stage(stage):
                        period_obj = p
                        break
                except KeyError:
                    continue
    else:
        if period_idx == "first":
            period_idx = 0
        try:
            # Try ModelCircuit.get_period() if it exists, else index directly
            period_obj = (model.get_period(period_idx)   # type: ignore[attr-defined]
                          if hasattr(model, "get_period")
                          else periods[period_idx])
        except (IndexError, KeyError):
            return None

    if period_obj is None or not hasattr(period_obj, "get_stage"):
        return None

    # ---------------------------------------------------------------------
    # 2) Locate the Stage
    # ---------------------------------------------------------------------
    try:
        stage_obj = period_obj.get_stage(stage)
    except (AttributeError, KeyError):
        return None

    # Helper that tries to pull an ndarray from a “solution-like” container
    def _extract_from_solution(sol_obj: Any) -> Optional[np.ndarray]:
        if sol_obj is None:
            return None
        if hasattr(sol_obj, sol_attr):
            policy_obj = getattr(sol_obj, sol_attr)
        elif isinstance(sol_obj, dict) and sol_attr in sol_obj:
            policy_obj = sol_obj[sol_attr]
        else:
            return None

        if key in ("", None):
            if isinstance(policy_obj, np.ndarray):
                return policy_obj
            return None

        if hasattr(policy_obj, key):
            return np.asarray(getattr(policy_obj, key))
        if isinstance(policy_obj, dict) and key in policy_obj:
            return np.asarray(policy_obj[key])
        return None

    # ---------------------------------------------------------------------
    # 3) Try dcsn.sol → policy
    # ---------------------------------------------------------------------
    if hasattr(stage_obj, "dcsn") and hasattr(stage_obj.dcsn, "sol"):
        arr = _extract_from_solution(stage_obj.dcsn.sol)
        if arr is not None:
            return arr

    # ---------------------------------------------------------------------
    # 4) Fallback: cntn.sol → policy
    # ---------------------------------------------------------------------
    if hasattr(stage_obj, "cntn") and hasattr(stage_obj.cntn, "sol"):
        arr = _extract_from_solution(stage_obj.cntn.sol)
        if arr is not None:
            return arr

    # ---------------------------------------------------------------------
    # 5) Last-ditch: attribute directly on the model
    # ---------------------------------------------------------------------
    try:
        return np.asarray(getattr(model, key))
    except AttributeError:
        return None



def make_policy_dev_metric(
    policy_attr: str,
    norm: Literal["L2", "Linf"],
    stage: str = "OWNC",
    sol_attr: str = "policy",
) -> Callable[[Any, CircuitRunner, np.ndarray], float]:
    """
    Factory to create policy deviation metrics.

    Args:
        policy_attr: Policy key/attribute name (e.g., "c", "a", "v")
        norm: Norm type - "L2" for Euclidean norm, "Linf" for infinity norm
        stage: Stage name to extract from (default: "OWNC")
        sol_attr: Solution attribute (default: "policy")

    Returns:
        Metric function that accepts (model, *, _runner, _x) and returns float

    Note:
        The L2 norm returns the raw Euclidean norm (not normalized).
        For RMS error, divide by sqrt(n) where n is the number of elements.
    """

    def metric(
        model: Any, *, _runner: Optional[CircuitRunner] = None, _x: Optional[np.ndarray] = None
    ) -> float:
        """Compute deviation of model policy from reference."""
        # Handle backward compatibility
        if _runner is None or _x is None:
            return np.nan

        # Get metric requirements for selective loading
        metric_name = f"dev_{policy_attr}_{norm}"
        periods_to_load, stages_to_load = get_metric_requirements([metric_name])
        metric_requirements = {
            'periods_to_load': periods_to_load,
            'stages_to_load': stages_to_load
        }
        
        # Load reference model with selective loading
        ref_model = load_reference_model(_runner, _x, metric_requirements)
        if ref_model is None:
            return np.nan

        # Get policies using the extractor
        policy = _extract_policy(model, stage, sol_attr, policy_attr)
        ref_policy = _extract_policy(ref_model, stage, sol_attr, policy_attr)

        #print(policy)

        if policy is None or ref_policy is None:
            return np.nan

        # Compute difference
        diff = policy - ref_policy

        # Return appropriate norm
        if norm == "L2":
            return float(np.linalg.norm(diff.ravel(), ord=2))
        else:  # Linf
            return float(np.linalg.norm(diff.ravel(), ord=np.inf))

    # Set a meaningful name for the metric
    metric.__name__ = f"dev_{policy_attr}_{norm}"
    metric.__doc__ = f"{norm} deviation of {policy_attr} from reference model"

    return metric


# Create concrete metric functions for consumption only
# (most reliable since consumption is commonly available)
dev_c_L2 = make_policy_dev_metric("c", "L2")
dev_c_Linf = make_policy_dev_metric("c", "Linf")

# These require specific model structures and may not work for all models
# Users should create their own with appropriate stage/sol_attr parameters
dev_a_L2 = make_policy_dev_metric("a", "L2")
dev_a_Linf = make_policy_dev_metric("a", "Linf")
dev_v_L2 = make_policy_dev_metric("v", "L2", sol_attr="value")
dev_v_Linf = make_policy_dev_metric("v", "Linf", sol_attr="value")
dev_pol_L2 = make_policy_dev_metric("pol", "L2")
dev_pol_Linf = make_policy_dev_metric("pol", "Linf")
