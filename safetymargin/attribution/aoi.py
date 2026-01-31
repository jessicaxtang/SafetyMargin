from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from ..models.base import BaseModel

"""
Add-One-In (AOI) attribution.

alpha_i(B) = R(Q, B ∪ {s_i}) - R(Q, B)

- Q: question/prompt stem
- M: minimal baseline (system/instructions)
- S: list of specification spans {s_i}
- A: intent additions {a_i} (used in pipeline.py and scenarios.py)
- Baselines B can be:
  * empty: []
  * minimal: M
  * targeted: M + subset T ⊂ S \ {s_i}
  * loo-equivalent: (M ∪ S) \ {s_i}
R can be: log-likelihood to refs, a reward/judge score in [0,1], etc.
We only require a scalar.
"""

@dataclass
class AOIConfig:
    normalize_within_run: bool = True
    temperatures: Tuple[float, ...] = (0.0,)  # average over temps for stability if model supports it
    n_samples: int = 1  # average over stochastic samples if model supports it
    targeted_pairs: Optional[List[Tuple[int, int]]] = None
    # If provided, compute alpha_i(M ∪ {s_j}) for each (i, j) pair (i != j)
    compute_loo_equivalent: bool = True  # alpha_i(C \ {s_i}) == LOO
    # Names for standard baselines
    include_empty_baseline: bool = True
    include_minimal_baseline: bool = True

class AOIAttributor:
    """
    Public API mirrors LOOAttributor:
      .attribute(Q, M, S, model) -> dict
    """

    def __init__(self, cfg: Optional[AOIConfig] = None):
        self.cfg = cfg or AOIConfig()

    def run(self, *args, **kwargs):  # type: ignore[override]
        raise NotImplementedError("AOIAttributor.run is not implemented; call attribute().")

    def _score(self, model: BaseModel, Q: str, context: List[str]) -> float:
        """
        Returns scalar success score ℛ(Q, context).
        The model wrapper may return:
          - a float directly, or
          - a dict with {"score": float, ...}
        """
        out = model.score(Q=Q, context=context)
        if isinstance(out, dict):
            return float(out.get("score"))
        return float(out)

    def _mean_score(self, model: BaseModel, Q: str, context: List[str]) -> float:
        # Average over temperatures/samples if supported; otherwise single call
        scores = []
        for _ in range(max(1, self.cfg.n_samples)):
            s = self._score(model, Q, context)
            scores.append(s)
        return float(np.mean(scores)) if len(scores) > 1 else scores[0]

    def _build_context(self, M: List[str], spans: Iterable[str]) -> List[str]:
        return list(M) + list(spans)

    def attribute(
        self,
        Q: str,
        M: List[str],
        S: List[str],
        model: BaseModel,
        # Optional: full context C to save callers a join
        C: Optional[List[str]] = None,
        # Optional targeted baseline sets T_k (each T is a list of indices into S excluding i)
        targeted_sets: Optional[List[List[int]]] = None,
    ) -> Dict[str, Any]:
        """
        Compute AOI across baselines for each span s_i.

        Returns a dict:
        {
          "config": {...},
          "spans": S,
          "baselines": ["empty","minimal","targeted:...","loo_equiv"],
          "alpha": { i: { baseline_name: float } },
          "alpha_matrix_targeted": optional np.ndarray [n_spans, n_spans] (i,j) for M∪{s_j},
          "context_scores": {"empty": float, "minimal": float, ...},  # baseline scores without s_i
        }
        """
        n = len(S)
        result: Dict[str, Any] = {
            "config": vars(self.cfg),
            "spans": S,
            "alpha": {},
            "baselines": [],
            "context_scores": {},
        }

        # Precompute canonical baselines (without s_i)
        baselines: List[Tuple[str, List[str]]] = []

        if self.cfg.include_empty_baseline:
            B_empty = []  # Q only
            baselines.append(("empty", B_empty))

        if self.cfg.include_minimal_baseline:
            B_min = list(M)
            baselines.append(("minimal", B_min))

        # Targeted baselines: M ∪ T where T ⊂ S\{s_i}
        # If targeted_sets not given, derive from cfg.targeted_pairs (i,j)
        targeted_pairs = self.cfg.targeted_pairs or []
        targeted_sets = targeted_sets or []
        # Canonicalize any provided targeted sets as unique names
        for T in targeted_sets:
            T = sorted(set(int(t) for t in T if 0 <= t < n))
            name = "targeted:T=" + ",".join(map(str, T))
            baselines.append((name, self._build_context(M, [S[t] for t in T])))

        # Pairwise targets M ∪ {s_j} (useful for interaction heatmaps)
        pairwise_names = []
        for (i_idx, j_idx) in targeted_pairs:
            if i_idx == j_idx or not (0 <= i_idx < n and 0 <= j_idx < n):
                continue
            name = f"pair:M+{{{j_idx}}}"
            # We'll expand per-i later; keep a marker for pairwise computation
            pairwise_names.append((i_idx, j_idx, name))

        # LOO-equivalent baseline per i: (M ∪ S) \ {s_i}
        if self.cfg.compute_loo_equivalent:
            # We'll add per-i baseline under the name "loo_equiv"
            pass

        # Compute baseline scores *once* where possible (not per-i)
        # For pairwise and loo-equivalent we compute per-i below.
        for bname, B in baselines:
            # Score baseline alone (without any s_i added)
            base_score = self._mean_score(model, Q, B)
            result["context_scores"][bname] = base_score

        # Main AOI: for each i, compute alpha_i(B) for all common baselines
        for i in range(n):
            s_i = S[i]
            result["alpha"][i] = {}
            for bname, B in baselines:
                with_i = list(B) + [s_i]
                score_with = self._mean_score(model, Q, with_i)
                score_base = result["context_scores"][bname]
                result["alpha"][i][bname] = float(score_with - score_base)

            # LOO-equivalent baseline for this i: (M ∪ S) \ {s_i}
            if self.cfg.compute_loo_equivalent:
                C_full = self._build_context(M, S)
                C_minus_i = self._build_context(M, [S[k] for k in range(n) if k != i])
                base_score = self._mean_score(model, Q, C_minus_i)
                with_i = list(C_full)  # = (C_minus_i ∪ {s_i})
                score_with = self._mean_score(model, Q, with_i)
                result["alpha"][i]["loo_equiv"] = float(score_with - base_score)

        # Optional pairwise matrix alpha_i(M ∪ {s_j})
        if len(targeted_pairs) > 0:
            mat = np.zeros((n, n), dtype=float)
            for (i_idx, j_idx, _nm) in pairwise_names:
                # Baseline is M ∪ {s_j} (no s_i yet)
                B = self._build_context(M, [S[j_idx]])
                base_score = self._mean_score(model, Q, B)
                score_with = self._mean_score(model, Q, B + [S[i_idx]])
                delta = float(score_with - base_score)
                try:
                    mat[i_idx, j_idx] = delta
                except TypeError:
                    mat[i_idx][j_idx] = delta
            result["alpha_matrix_targeted"] = mat

        # Optional normalization (z-score) per baseline across spans for interpretability
        if self.cfg.normalize_within_run:
            for bname in list(result["context_scores"].keys()) + (["loo_equiv"] if self.cfg.compute_loo_equivalent else []):
                # collect alphas for this baseline
                vals = [result["alpha"][i][bname] for i in range(n) if bname in result["alpha"][i]]
                if len(vals) >= 2:
                    mu = float(np.mean(vals))
                    sd = float(np.std(vals) + 1e-8)
                    for i in range(n):
                        if bname in result["alpha"][i]:
                            result["alpha"][i][bname] = (result["alpha"][i][bname] - mu) / sd

        # Register baseline names for consumers
        names = [bname for (bname, _) in baselines]
        if self.cfg.compute_loo_equivalent:
            names.append("loo_equiv")
        result["baselines"] = names

        return result
