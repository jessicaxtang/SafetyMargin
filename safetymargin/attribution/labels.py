"""Decision labeling and repair suggestions for safety attribution."""

from statistics import median
from typing import Any, Dict, List, Tuple


def assign_decision_labels(
    spans: List[str],
    loo_vals: List[float],
    aoi: Dict[int, Dict[str, float]],
    delta: Dict[Tuple[int, int], float],
    tau_N: float = 0.2,
    tau_S: float = 0.2,
    tau_zero: float = 0.05,
) -> Dict[int, List[str]]:
    """
    Assign decision labels to each span based on LOO/AOI values.
    
    Labels include:
    - Safety-helping (keep/strengthen): span improves safety
    - Risk-increasing (candidate for removal): span harms safety
    - Neutral: minimal impact
    - Conflict/Jailbreak (risk on M): span harmful when added to minimal context
    - Shortcut suspicion: span depends on interaction with another span
    
    Args:
        spans: List of specification span texts
        loo_vals: LOO necessity scores for each span
        aoi: AOI sufficiency scores, indexed by span and baseline
        delta: Pairwise interaction gains
        tau_N: Threshold for LOO necessity
        tau_S: Threshold for AOI sufficiency
        tau_zero: Threshold for near-zero effects
        
    Returns:
        Dictionary mapping span index to list of label strings
    """
    labels: Dict[int, List[str]] = {}
    
    for i, span in enumerate(spans):
        span_labels: List[str] = []
        
        loo_val = loo_vals[i]
        a_min = aoi.get(i, {}).get("M")
        a_empty = aoi.get(i, {}).get("∅")
        
        # Primary necessity/sufficiency label
        if abs(loo_val) < tau_N and (a_min is not None and abs(a_min) < tau_S):
            span_labels.append("Neutral")
        else:
            if loo_val > tau_N:
                span_labels.append("Safety-helping (keep/strengthen)")
            elif loo_val < -tau_N:
                span_labels.append("Risk-increasing (candidate for removal)")
        
        # Conflict/jailbreak signal - span harmful when added to minimal
        if a_min is not None and a_min < -tau_S:
            span_labels.append("Conflict/Jailbreak (risk on M)")
        
        # Shortcut detection (dependency on other spans)
        if a_min is not None and abs(a_min) <= tau_zero:
            for key in aoi.get(i, {}).keys():
                if key.startswith("M+{s_") and aoi[i][key] > tau_S:
                    j = int(key.split("_")[1].rstrip("}"))
                    span_labels.append(f"Shortcut suspicion (collapses with s_{j})")
        
        labels[i] = span_labels
    
    return labels


def generate_repair_suggestions(
    spans: List[str],
    loo_vals: List[float],
    aoi: Dict[int, Dict[str, float]],
    intent_aoi: Dict[str, float],
    tau_N: float = 0.2,
    tau_S: float = 0.2,
    tau_I: float = 0.15,
) -> Dict[str, List[str]]:
    """
    Generate repair suggestions based on attribution scores.
    
    Uses robust z-scores (MAD-based) to detect outliers and anomalies.
    
    Args:
        spans: List of specification span texts
        loo_vals: LOO necessity scores
        aoi: AOI sufficiency scores
        intent_aoi: AOI scores for intent additions
        tau_N: Threshold for LOO necessity
        tau_S: Threshold for AOI sufficiency
        tau_I: Threshold for intent AOI
        
    Returns:
        Dictionary with keys:
        - "strong": High-confidence removal/rewrite suggestions
        - "review": Human review flags
    """
    # Compute robust z-scores using MAD (median absolute deviation)
    med = median(loo_vals) if loo_vals else 0.0
    mad = median([abs(v - med) for v in loo_vals]) if loo_vals else 1.0
    # Scale factor to match standard deviation for normal distribution
    sigma_null = max(mad * 1.4826, 1e-6)
    
    strong: List[str] = []
    review: List[str] = []
    
    for i, span in enumerate(spans):
        loo_val = loo_vals[i]
        aoi_min = aoi.get(i, {}).get("M")
        z_val = (loo_val - med) / sigma_null if sigma_null else 0.0
        
        # Risk flags
        risk_loo_strong = (loo_val < -tau_N) or (z_val < -3)
        risk_loo_soft = (loo_val < -tau_N) or (z_val < -2)
        risk_aoi = aoi_min is not None and aoi_min < -tau_S
        
        # Strong recommendation: both LOO and AOI indicate harm
        if risk_loo_strong and risk_aoi:
            strong.append(
                f"Remove or rewrite s_{i} ({span}) "
                f"[LOO={loo_val:.6f} (z={z_val:.2f}); AOI(M)={aoi_min:.6f} < -τ_S] "
                f"and re-run to ensure f(C) increases (safer)."
            )
        # Soft recommendation: one signal indicates potential harm
        elif risk_loo_soft or risk_aoi:
            parts = []
            if risk_loo_soft:
                parts.append(f"LOO={loo_val:.6f} (z={z_val:.2f})")
            if risk_aoi:
                parts.append(f"AOI(M)={aoi_min:.6f} < -τ_S")
            review.append(
                f"Flag s_{i} ({span}) for human review [{'; '.join(parts)}]"
            )
    
    # Intent suggestions
    if intent_aoi:
        intents_sorted = sorted(intent_aoi.items(), key=lambda kv: abs(kv[1]), reverse=True)
        if intents_sorted:
            top_key, top_val = intents_sorted[0]
            if top_val < -tau_I:
                strong.append(
                    f"Counter or weaken '{top_key}' which appears risk-increasing "
                    f"(AOI={top_val:.6f}). Re-run to verify improvement."
                )
    
    return {"strong": strong, "review": review}


def compute_robust_sigma(loo_vals: List[float]) -> float:
    """
    Compute robust standard deviation estimate using MAD.
    
    Args:
        loo_vals: LOO necessity values
        
    Returns:
        Robust sigma estimate
    """
    if not loo_vals:
        return 1.0
    med = median(loo_vals)
    mad = median([abs(v - med) for v in loo_vals])
    return max(mad * 1.4826, 1e-6)
