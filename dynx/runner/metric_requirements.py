"""
Metric requirements specification for selective loading.

This module defines which periods and stages each metric needs to access,
enabling selective loading of only the required data.
"""

from typing import Dict, List, Optional, Tuple

# Metric requirements mapping
# Format: metric_name -> (periods_needed, stages_per_period)
METRIC_REQUIREMENTS: Dict[str, Tuple[List[int], Dict[int, Optional[List[str]]]]] = {
    # Euler error needs period 0 (OWNC) and period 1 (all stages)
    "euler_error": ([0, 1], {0: ["OWNC"], 1: None}),
    
    # Deviation metrics only need period 0 OWNC stage for comparison
    "dev_c_L2": ([0], {0: ["OWNC"]}),
    "dev_c_Linf": ([0], {0: ["OWNC"]}),
    "dev_a_L2": ([0], {0: ["OWNC"]}),
    "dev_a_Linf": ([0], {0: ["OWNC"]}),
    "dev_v_L2": ([0], {0: ["OWNC"]}),
    "dev_v_Linf": ([0], {0: ["OWNC"]}),
    "dev_pol_L2": ([0], {0: ["OWNC"]}),
    "dev_pol_Linf": ([0], {0: ["OWNC"]}),
    
    # Plotting metrics - EGM plots need OWNC from period 0
    "plot_egm": ([0], {0: ["OWNC"]}),
    
    # Policy plots need multiple stages from period 0
    "plot_policy": ([0], {0: ["TENU", "OWNH", "OWNC", "RNTH", "RNTC"]}),
    
    # Value/Q function comparison plots need OWNC from period 0
    "plot_value_q": ([0], {0: ["OWNC"]}),
    
    # Comparison plot metrics (from metrics.py)
    "plot_c_comparison": ([0], {0: ["OWNC"]}),
    "plot_v_comparison": ([0], {0: ["OWNC"]}),
    "plot_a_comparison": ([0], {0: ["OWNC"]}),
    "plot_h_comparison": ([0], {0: ["OWNH"]}),
}

def get_superset_requirements() -> Tuple[List[int], Dict[int, Optional[List[str]]]]:
    """
    Get the superset of all metric requirements.
    
    This returns the union of all periods and stages needed by any metric,
    ensuring a single cached model can serve all metrics. This is more
    efficient than caching different subsets for different metrics.
    
    Returns
    -------
    periods_to_load : List[int]
        [0, 1] - the superset of periods needed
    stages_to_load : Dict[int, Optional[List[str]]]
        Period 0: All stages used by any metric
        Period 1: None (all stages, required by euler_error)
    """
    return (
        [0, 1],  # Periods 0 and 1 cover all metrics
        {
            0: ['OWNC', 'OWNH', 'TENU', 'RNTH', 'RNTC'],  # Union of all period 0 stages
            1: None  # All stages for period 1 (required by euler_error)
        }
    )

def get_metric_requirements(metric_names: List[str]) -> Tuple[List[int], Dict[int, Optional[List[str]]]]:
    """
    Get combined requirements for a list of metrics.
    
    NOTE: With superset caching, this function is primarily used for 
    documentation and understanding individual metric needs. The actual
    caching uses get_superset_requirements() to ensure all metrics
    share the same cached model.
    
    Parameters
    ----------
    metric_names : List[str]
        List of metric names to get requirements for
        
    Returns
    -------
    periods_to_load : List[int]
        Combined list of periods needed
    stages_to_load : Dict[int, Optional[List[str]]]
        Combined mapping of stages needed per period
    """
    if not metric_names:
        return None, None
        
    all_periods = set()
    stages_by_period = {}
    
    for metric in metric_names:
        if metric in METRIC_REQUIREMENTS:
            periods, stages = METRIC_REQUIREMENTS[metric]
            all_periods.update(periods)
            
            for period, stage_list in stages.items():
                if period not in stages_by_period:
                    stages_by_period[period] = set() if stage_list is not None else None
                
                if stage_list is not None and stages_by_period[period] is not None:
                    stages_by_period[period].update(stage_list)
                elif stage_list is None:
                    # If any metric needs all stages for a period, mark it as None
                    stages_by_period[period] = None
    
    # Convert sets back to lists
    periods_to_load = sorted(list(all_periods))
    stages_to_load = {
        p: sorted(list(stages)) if stages is not None and isinstance(stages, set) else stages
        for p, stages in stages_by_period.items()
    }
    
    return periods_to_load, stages_to_load

def is_comparison_metric(metric_name: str) -> bool:
    """Check if a metric is a comparison metric that needs reference loading."""
    return metric_name.startswith(("dev_", "plot_"))