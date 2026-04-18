
"""Plot an example-level attribution heatmap for handcrafted prompts.

This script consumes the per-unit attribution CSV emitted by
`scripts/reference_margin_attribution.py` for the handcrafted prompt-unit
leave-one-out runs. Each cell displays the attribution value for a prompt unit
within an example. Blue indicates helpful/helpful contributions
(positive margin deltas), while red highlights harmful contributions.

Ground-truth labels (helpful vs harmful) are read from a small CSV so we can
overlay ✓/✗ markers to visualize agreement between the ground-truth judgment and
the measured attribution sign.
"""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

TAU_POS = 0.5 # Positive threshold for helpful
TAU_NEG = -0.5  # Negative threshold for harmful
METRIC_COLUMN_CANDIDATES = ["delta_ref_z"] #, "attribution"]

def get_default_paths(model_name: str):
    model_dir = f"experiments-basic-mar30/reference_attribution_n100_seed42_{model_name}"
    out_prefix = f"visualization/out_basic_mar30_{model_name}/handcrafted_prompt_heatmap"
    return {
        "per_unit": Path(model_dir) / "per_unit_rows.csv",
        "examples": Path(model_dir) / "examples.csv",
        "ground_truth": Path("dataset/basic/per_unit_rows_labelled.csv"),
        "output": Path(f"{out_prefix}.png"),
        "bucket_output": Path(f"{out_prefix}_bucket_matrix.png"),
        "bucket_csv": Path(f"{out_prefix}_bucket_rows.csv"),
        "scatter_output": Path(f"{out_prefix}_loo_aoi_scatter.png"),
    }


def select_metric_column(df: pd.DataFrame, candidates: list[str] | None = None) -> str:
    """Pick the first available metric column, preferring z-scored deltas."""
    candidates = candidates or METRIC_COLUMN_CANDIDATES
    for col in candidates:
        if col in df.columns:
            return col
    available = ", ".join(df.columns)
    raise ValueError(
        f"None of the metric columns {candidates} were found in the dataframe. "
        f"Available columns: {available}"
    )


def _shorten(text: str, width: int = 80) -> str:
    single_line = " ".join(str(text).split())
    if not single_line:
        return ""
    return textwrap.shorten(single_line, width=width, placeholder="…")


def load_per_unit(per_unit_path: Path, example_filter: set[int] | None) -> pd.DataFrame:
    df = pd.read_csv(per_unit_path)
    prompt_units = df[df["unit_kind"] == "prompt_unit"].copy()
    prompt_units["example_index"] = prompt_units["example_index"].astype(int)
    prompt_units["unit_index"] = prompt_units["unit_index"].astype(int)

    if example_filter is not None:
        prompt_units = prompt_units[prompt_units["example_index"].isin(example_filter)]

    if prompt_units.empty:
        raise ValueError(
            "No prompt units remain after filtering. Check the example indices or the per-unit CSV."
        )

    return prompt_units





def load_ground_truth_labels(ground_truth_path: Path, label_column: str | None) -> pd.DataFrame:
    if not ground_truth_path.exists():
        raise FileNotFoundError(
            "Ground-truth label file missing. Create it at "
            f"{ground_truth_path} with columns example_index,unit_index,ground_truth."
        )

    ground_truth_df = pd.read_csv(ground_truth_path, comment="#")
    label_candidates = [
        label_column,
        "ground_truth",
    ]
    label_col = next(
        (col for col in label_candidates if col and col in ground_truth_df.columns),
        None,
    )
    if label_col is None:
        raise ValueError(
            "Ground-truth label file must contain either the column specified via --ground-truth-column "
            "or 'ground_truth'."
        )

    required_cols = {"example_index", "unit_index", label_col}
    missing = required_cols - set(ground_truth_df.columns)
    if missing:
        raise ValueError(
            f"Ground-truth label file {ground_truth_path} is missing columns: {sorted(missing)}"
        )

    ground_truth_df["example_index"] = ground_truth_df["example_index"].astype(int)
    ground_truth_df["unit_index"] = ground_truth_df["unit_index"].astype(int)
    ground_truth_df["ground_truth"] = ground_truth_df[label_col].astype(str).str.strip().str.lower()

    # Allow 'helpful', 'harmful', and 'neutral' as valid labels
    valid_labels = {"helpful", "harmful", "neutral"}
    if not ground_truth_df["ground_truth"].isin(valid_labels).all():
        bad = ground_truth_df.loc[~ground_truth_df["ground_truth"].isin(valid_labels), ["example_index", "unit_index", label_col]]
        raise ValueError(
            "Ground-truth labels must be 'helpful', 'harmful', or 'neutral'. "
            f"Problem rows:\n{bad.to_string(index=False)}"
        )

    return ground_truth_df[["example_index", "unit_index", "ground_truth"]]





def attach_ground_truth_labels(per_unit: pd.DataFrame, ground_truth: pd.DataFrame) -> pd.DataFrame:
    per_unit_base = per_unit.drop(columns=["ground_truth"], errors="ignore")
    merged = per_unit_base.merge(
        ground_truth, on=["example_index", "unit_index"], how="left", validate="one_to_one"
    )
    missing = merged["ground_truth"].isna()
    if missing.any():
        missing_rows = merged.loc[
            missing, ["example_index", "unit_index", "unit_text"]
        ]
        raise ValueError(
            "Missing ground_truth labels for the following prompt units:\n"
            f"{missing_rows.to_string(index=False)}"
        )
    return merged

def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else np.nan

def sweep_tau(per_unit_ground_truth: pd.DataFrame, value_column: str = "attribution", tau_range=None):
    """
    Print metrics for a range of TAU values to help select a threshold.
    This version sweeps both tau_pos and tau_neg (symmetric).
    """
    if tau_range is None:
        tau_range = [x * 0.1 for x in range(0, 21)]  # 0.0 to 2.0 inclusive
    print("\nTAU sweep results (value_column='{}'):".format(value_column))
    print("TAU\tAgreement\tHelpfulPrec\tHarmfulPrec\tNeutralCt\tCoverage")
    global TAU_POS, TAU_NEG
    orig_tau_pos = TAU_POS
    orig_tau_neg = TAU_NEG
    total_n = len(per_unit_ground_truth)
    for tau in tau_range:
        TAU_POS = tau
        TAU_NEG = -tau
        metrics = compute_metrics(per_unit_ground_truth, value_column, tau_pos=TAU_POS, tau_neg=TAU_NEG)
        neutral_ct = metrics.get('neutral_count', 0)
        coverage = 1.0 - (neutral_ct / total_n) if total_n > 0 else float('nan')
        print(f"{tau:.2f}\t{metrics.get('agreement_rate', float('nan')):.3f}\t"
              f"{metrics.get('helpful_precision', float('nan')):.3f}\t"
              f"{metrics.get('harmful_precision', float('nan')):.3f}\t"
              f"{neutral_ct}\t{coverage:.3f}")
    TAU_POS = orig_tau_pos
    TAU_NEG = orig_tau_neg
    
def sweep_tau1(per_unit_ground_truth, value_column="attribution"):
    import numpy as np
    taus = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    rows = []
    for tau in taus:
        sub = per_unit_ground_truth.dropna(subset=[value_column, "ground_truth"]).copy()
        x = sub[value_column].astype(float)

        sub["pred"] = np.where(x >= tau, "helpful",
                        np.where(x <= -tau, "harmful", "neutral"))

        covered = sub[sub["pred"] != "neutral"].copy()
        coverage = len(covered) / len(sub) if len(sub) else np.nan
        agreement = (covered["pred"] == covered["ground_truth"]).mean() if len(covered) else np.nan

        rows.append({
            "tau": tau,
            "coverage": coverage,
            "agreement_on_covered": agreement,
            "num_covered": len(covered),
            "num_neutral": int((sub["pred"] == "neutral").sum()),
        })
    return rows


def sweep_two_thresholds(per_unit_ground_truth: pd.DataFrame, value_column: str = "attribution",
                        tau_pos_list=None, tau_neg_list=None):
    """
    Print metrics for a grid of positive and negative thresholds.
    """
    if tau_pos_list is None:
        tau_pos_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    if tau_neg_list is None:
        tau_neg_list = [-0.25, -0.5, -0.75, -1.0, -1.5, -2.0]
    print("\ntau_neg\ttau_pos\tcoverage\thelpful_prec\thelpful_rec\tharmful_prec\tharmful_rec\tmacro_F1")
    total_n = len(per_unit_ground_truth)
    for tau_pos in tau_pos_list:
        for tau_neg in tau_neg_list:
            # Labeling logic: helpful >= tau_pos, harmful <= tau_neg, else neutral
            def label_with_two_thresholds(val):
                if val >= tau_pos:
                    return "helpful"
                elif val <= tau_neg:
                    return "harmful"
                else:
                    return "neutral"
            df = per_unit_ground_truth.copy()
            df["metric_value"] = df[value_column].astype(float)
            df["model_label"] = df["metric_value"].apply(label_with_two_thresholds)
            df["agreement"] = (df["model_label"] == df["ground_truth"]) & (df["model_label"] != "neutral")
            tp = ((df["model_label"] == "helpful") & (df["ground_truth"] == "helpful")).sum()
            fp = ((df["model_label"] == "helpful") & (df["ground_truth"] == "harmful")).sum()
            tn = ((df["model_label"] == "harmful") & (df["ground_truth"] == "harmful")).sum()
            fn = ((df["model_label"] == "harmful") & (df["ground_truth"] == "helpful")).sum()
            neutral_ct = (df["model_label"] == "neutral").sum()
            coverage = 1.0 - (neutral_ct / total_n) if total_n > 0 else float('nan')
            # Precision, recall, F1
            helpful_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            helpful_rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            harmful_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
            harmful_rec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            # Macro F1
            def f1(p, r):
                return 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            f1_helpful = f1(helpful_prec, helpful_rec)
            f1_harmful = f1(harmful_prec, harmful_rec)
            macro_f1 = 0.5 * (f1_helpful + f1_harmful)
            print(f"{tau_neg:.2f}\t{tau_pos:.2f}\t{coverage:.3f}\t{helpful_prec:.3f}\t{helpful_rec:.3f}\t{harmful_prec:.3f}\t{harmful_rec:.3f}\t{macro_f1:.3f}")



def compute_metrics(per_unit_ground_truth: pd.DataFrame, value_column: str, tau_pos: float = None, tau_neg: float = None) -> dict[str, float]:
    if value_column not in per_unit_ground_truth.columns:
        raise ValueError(f"Column '{value_column}' not found in per-unit dataframe.")

    df = per_unit_ground_truth.dropna(subset=[value_column]).copy()
    metrics: dict[str, float] = {
        "value_column": value_column,
        "total_units": float(len(df)),
    }
    if df.empty:
        metrics.update(
            {
                "agreement_rate": np.nan,
                "helpful_precision": np.nan,
                "helpful_recall": np.nan,
                "harmful_precision": np.nan,
                "harmful_recall": np.nan,
                "mean_abs_attr_helpful": np.nan,
                "mean_abs_attr_harmful": np.nan,
            }
        )
        return metrics

    if tau_pos is None:
        tau_pos = TAU_POS
    if tau_neg is None:
        tau_neg = TAU_NEG

    df["metric_value"] = df[value_column].astype(float)
    # Threshold for neutral: helpful >= tau_pos, harmful <= tau_neg, else neutral
    def label_with_threshold(val):
        if val >= tau_pos:
            return "helpful"
        elif val <= tau_neg:
            return "harmful"
        else:
            return "neutral"
    df["model_label"] = df["metric_value"].apply(label_with_threshold)
    # Only compute agreement for non-neutral model labels
    df["agreement"] = (df["model_label"] == df["ground_truth"]) & (df["model_label"] != "neutral")

    tp = ((df["model_label"] == "helpful") & (df["ground_truth"] == "helpful")).sum()
    fp = ((df["model_label"] == "helpful") & (df["ground_truth"] == "harmful")).sum()
    tn = ((df["model_label"] == "harmful") & (df["ground_truth"] == "harmful")).sum()
    fn = ((df["model_label"] == "harmful") & (df["ground_truth"] == "helpful")).sum()

    metrics.update(
        {
            "agreement_rate": float(df.loc[df["model_label"] != "neutral", "agreement"].mean()),
            "helpful_precision": _safe_div(tp, tp + fp),
            "helpful_recall": _safe_div(tp, tp + fn),
            "harmful_precision": _safe_div(tn, tn + fn),
            "harmful_recall": _safe_div(tn, tn + fp),
            "neutral_count": int((df["model_label"] == "neutral").sum()),
        }
    )

    abs_attr = df.assign(abs_attr=df["metric_value"].abs())
    by_ground_truth = abs_attr.groupby("ground_truth")["abs_attr"].mean().to_dict()
    for label, value in by_ground_truth.items():
        metrics[f"mean_abs_attr_{label}"] = float(value)

    return metrics


def print_metrics(metrics: dict[str, float], label: str) -> None:
    total = int(metrics.get("total_units", 0))
    print(f"{label} metrics")
    print(f"  Total prompt units: {total}")
    agreement = metrics.get("agreement_rate")
    if agreement is None or np.isnan(agreement):
        print("  No measurements available.")
        return
    print(f"  Agreement rate: {agreement:.3f}")
    print(
        "  Helpful precision/recall: "
        f"{metrics['helpful_precision']:.3f} / {metrics['helpful_recall']:.3f}"
    )
    print(
        "  Harmful precision/recall: "
        f"{metrics['harmful_precision']:.3f} / {metrics['harmful_recall']:.3f}"
    )
    print(f"  Neutral count: {metrics.get('neutral_count', 0)}")
    helpful_abs = metrics.get("mean_abs_attr_helpful")
    harmful_abs = metrics.get("mean_abs_attr_harmful")
    if helpful_abs is not None or harmful_abs is not None:
        help_text = "nan" if helpful_abs is None else f"{helpful_abs:.3f}"
        harm_text = "nan" if harmful_abs is None else f"{harmful_abs:.3f}"
        print(
            "  Mean |attribution| by ground_truth class: "
            f"helpful={help_text}, harmful={harm_text}"
        )


def _label_from_value(value: float, tau_pos: float = None, tau_neg: float = None) -> str:
    if tau_pos is None:
        tau_pos = TAU_POS
    if tau_neg is None:
        tau_neg = TAU_NEG
    if value >= tau_pos:
        return "helpful"
    if value <= tau_neg:
        return "harmful"
    return "neutral"


def annotate_buckets(
    per_unit_ground_truth: pd.DataFrame,
    tau_pos: float = None,
    tau_neg: float = None,
    value_column: str = "attribution",
) -> pd.DataFrame:
    if "aoi_delta" not in per_unit_ground_truth.columns:
        return per_unit_ground_truth.copy()

    df = per_unit_ground_truth.copy()
    buckets: list[str] = []
    for row in df.itertuples():
        loo_value = getattr(row, value_column, np.nan)
        aoi_value = getattr(row, "aoi_delta", np.nan)
        if pd.isna(loo_value) or pd.isna(aoi_value):
            buckets.append("")
            continue
        loo_label = _label_from_value(loo_value, tau_pos, tau_neg)
        aoi_label = _label_from_value(aoi_value, tau_pos, tau_neg)
        bucket = ""
        if loo_label == "helpful":
            if aoi_label in {"helpful", "neutral"}:
                bucket = "HH"
            elif aoi_label == "harmful":
                bucket = "HL"
        elif loo_label == "harmful":
            if aoi_label in {"helpful", "neutral"}:
                bucket = "LH"
            elif aoi_label == "harmful":
                bucket = "LL"
        buckets.append(bucket)
    df["loo_aoi_bucket"] = buckets
    return df


def save_bucket_csv(per_unit: pd.DataFrame, output_path: Path) -> None:
    if "loo_aoi_bucket" not in per_unit.columns:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_unit.to_csv(output_path, index=False)


def compute_bucket_counts(per_unit_ground_truth: pd.DataFrame) -> dict[str, int] | None:
    if "loo_aoi_bucket" not in per_unit_ground_truth.columns:
        return None

    counts = {"HH": 0, "HL": 0, "LH": 0, "LL": 0}
    for bucket in per_unit_ground_truth["loo_aoi_bucket"]:
        if bucket in counts:
            counts[bucket] += 1
    return counts


def print_bucket_summary(counts: dict[str, int]) -> None:
    total = sum(counts.values())
    if total == 0:
        print("LOO vs AOI buckets\n  No overlapping LOO/AOI rows to summarize.")
        return
    print("LOO vs AOI buckets")
    for label in ["HH", "HL", "LH", "LL"]:
        count = counts.get(label, 0)
        pct = (count / total) * 100 if total else 0.0
        desc = {
            "HH": "LOO helpful & AOI helpful/neutral",
            "HL": "LOO helpful & AOI harmful",
            "LH": "LOO harmful & AOI helpful/neutral",
            "LL": "LOO harmful & AOI harmful",
        }[label]
        print(f"  {label}: {count} ({pct:.1f}%) – {desc}")


def plot_bucket_matrix(counts: dict[str, int], output_path: Path) -> None:
    total = sum(counts.values())
    if total == 0:
        return
    matrix = np.array(
        [
            [counts.get("HH", 0), counts.get("HL", 0)],
            [counts.get("LH", 0), counts.get("LL", 0)],
        ],
        dtype=float,
    )
    perc_matrix = (matrix / total) * 100
    labels = np.empty_like(matrix, dtype=object)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            labels[i, j] = f"{int(matrix[i, j])}\n({perc_matrix[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(4, 3.6))
    sns.heatmap(
        perc_matrix,
        annot=labels,
        fmt="",
        cmap="Blues",
        cbar_kws={"label": "% of spans"},
        xticklabels=["AOI helpful/neutral", "AOI harmful"],
        yticklabels=["LOO helpful", "LOO harmful"],
        ax=ax,
    )
    ax.set_title("LOO vs AOI bucket matrix")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_scatter(per_unit_ground_truth: pd.DataFrame, output_path: Path, value_column: str) -> None:
    if "aoi_delta" not in per_unit_ground_truth.columns:
        return
    if value_column not in per_unit_ground_truth.columns:
        return
    data = per_unit_ground_truth.dropna(subset=[value_column, "aoi_delta"]).copy()
    if data.empty:
        return

    color_map = {"helpful": "#1f78b4", "harmful": "#e31a1c"}
    data["ground_truth"] = data["ground_truth"].fillna("neutral")
    colors = data["ground_truth"].map(color_map).fillna("#6c757d")

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(
        data[value_column],
        data["aoi_delta"],
        c=colors,
        alpha=0.7,
        edgecolors="none",
    )
    ax.axvline(0, color="#bbbbbb", linewidth=1, linestyle="--")
    ax.axhline(0, color="#bbbbbb", linewidth=1, linestyle="--")
    ax.set_xlabel(f"ΔM_LOO ({value_column})")
    ax.set_ylabel("ΔM_AOI (add-one-in delta)")
    ax.set_title("LOO vs AOI margin scatter")

    handles = []
    labels = []
    for label, color in color_map.items():
        handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color, markersize=8))
        labels.append(label.capitalize())
    handles.append(plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#6c757d", markersize=8))
    labels.append("Neutral")
    ax.legend(handles, labels, title="Ground-truth label", loc="best")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)



def plot_attribution_histogram(per_unit: pd.DataFrame, output_path: Path, value_column: str) -> None:
    if value_column not in per_unit.columns:
        print(f"No '{value_column}' column found in per-unit data.")
        return

    data = per_unit[value_column].dropna()
    if data.empty:
        print("No attribution values to plot.")
        return

    # Use shared bin edges for every histogram layer so stacked colors align.
    bins = 40
    data_min = float(data.min())
    data_max = float(data.max())
    if data_min == data_max:
        # Expand a degenerate range slightly to avoid matplotlib warnings.
        half_width = 0.5 if data_min == 0 else abs(data_min) * 0.01
        bin_edges = np.linspace(data_min - half_width, data_max + half_width, bins + 1)
    else:
        bin_edges = np.linspace(data_min, data_max, bins + 1)

    if "ground_truth" not in per_unit.columns:
        print("No 'ground_truth' column found in per-unit data. Plotting single-color histogram.")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=bin_edges, color="#1f78b4", edgecolor="white", alpha=0.85)
    else:
        # Color code by ground truth
        gt_classes = ["helpful", "harmful", "neutral"]
        colors = {"helpful": "#1f78b4", "harmful": "#e31a1c", "neutral": "#bbbbbb"}
        labels = {"helpful": "Helpful", "harmful": "Harmful", "neutral": "Neutral"}
        fig, ax = plt.subplots(figsize=(6, 4))
        for gt in gt_classes:
            subset = per_unit[per_unit["ground_truth"] == gt][value_column].dropna()
            if not subset.empty:
                ax.hist(
                    subset,
                    bins=bin_edges,
                    color=colors[gt],
                    edgecolor="white",
                    alpha=0.75,
                    label=labels[gt],
                )
    ax.axvline(0, color="#bbbbbb", linewidth=1, linestyle="--", label="Zero")
    ax.axvline(TAU_POS, color="#43a047", linewidth=1.5, linestyle=":", label=f"+TAU ({TAU_POS})")
    ax.axvline(TAU_NEG, color="#e53935", linewidth=1.5, linestyle=":", label=f"-TAU ({TAU_NEG})")
    ax.set_xlabel(f"{value_column} value")
    ax.set_ylabel("Count")
    ax.set_title(f"Distribution of {value_column}")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name",
        type=str,
        choices=["llama-3-8b", "llama-3.2-1b", "llama-3.2-3b"],
        default="llama-3-1b",
        help="Model name to select default paths for.",
    )
    # Placeholders, will be replaced after parsing
    parser.add_argument(
        "--per-unit-path",
        type=Path,
        default=None,
        help="CSV containing per-unit attributions (default: depends on --model-name).",
    )
    parser.add_argument(
        "--examples-path",
        type=Path,
        default=None,
        help="CSV with per-example metadata to construct labels.",
    )
    parser.add_argument(
        "--ground-truth-path",
        type=Path,
        default=None,
        help="CSV that stores ground-truth helpful/harmful labels for each prompt unit.",
    )
    parser.add_argument(
        "--ground-truth-column",
        type=str,
        default=None,
        help="Column name to use for ground-truth labels (defaults to ground_truth if present).",
    )
    parser.add_argument(
        "--example-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional subset of example indices to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to write the rendered heatmap PNG.",
    )
    parser.add_argument(
        "--bucket-output",
        type=Path,
        default=None,
        help="Where to write the LOO vs AOI bucket matrix PNG.",
    )
    parser.add_argument(
        "--bucket-csv",
        type=Path,
        default=None,
        help="CSV to write per-unit rows annotated with LOO/AOI bucket labels.",
    )
    parser.add_argument(
        "--scatter-output",
        type=Path,
        default=None,
        help="Where to write the LOO vs AOI scatter plot.",
    )

    args = parser.parse_args()

    # Set defaults based on model size if not provided
    defaults = get_default_paths(args.model_name)
    if args.per_unit_path is None:
        args.per_unit_path = defaults["per_unit"]
    if args.examples_path is None:
        args.examples_path = defaults["examples"]
    if args.ground_truth_path is None:
        args.ground_truth_path = defaults["ground_truth"]
    if args.output is None:
        args.output = defaults["output"]
    if args.bucket_output is None:
        args.bucket_output = defaults["bucket_output"]
    if args.bucket_csv is None:
        args.bucket_csv = defaults["bucket_csv"]
    if args.scatter_output is None:
        args.scatter_output = defaults["scatter_output"]
    return args


def main() -> None:
    args = parse_args()
    example_filter = set(args.example_indices) if args.example_indices else None

    per_unit = load_per_unit(args.per_unit_path, example_filter)
    ground_truth = load_ground_truth_labels(args.ground_truth_path, args.ground_truth_column)
    per_unit_ground_truth = attach_ground_truth_labels(per_unit, ground_truth)
    metric_column = select_metric_column(per_unit_ground_truth)

    # sweep_tau here
    sweep_tau(per_unit_ground_truth, value_column=metric_column)

    sweep_two_thresholds(per_unit_ground_truth, value_column=metric_column)

    per_unit_ground_truth = annotate_buckets(
        per_unit_ground_truth,
        tau_pos=TAU_POS,
        tau_neg=TAU_NEG,
        value_column=metric_column,
    )

    loo_metrics = compute_metrics(per_unit_ground_truth, value_column=metric_column)
    print_metrics(loo_metrics, label="LOO attribution")

    if "aoi_delta" in per_unit_ground_truth.columns:
        aoi_metrics = compute_metrics(per_unit_ground_truth, value_column="aoi_delta")
        if aoi_metrics["total_units"] > 0:
            print_metrics(aoi_metrics, label="AOI delta")
        else:
            print("AOI delta metrics\n  Column present but no finite AOI rows to score.")
    else:
        print("AOI delta metrics\n  Column 'aoi_delta' not found in per-unit CSV; skipping AOI report.")

    bucket_counts = compute_bucket_counts(per_unit_ground_truth)
    if bucket_counts is not None:
        print_bucket_summary(bucket_counts)
        plot_bucket_matrix(bucket_counts, args.bucket_output)
        save_bucket_csv(per_unit_ground_truth, args.bucket_csv)
        plot_scatter(per_unit_ground_truth, args.scatter_output, value_column=metric_column)
        print(f"Saved LOO vs AOI bucket matrix to {args.bucket_output}")
        print(f"Saved per-unit bucket annotations to {args.bucket_csv}")
        print(f"Saved LOO vs AOI scatter plot to {args.scatter_output}")

    histogram_path = args.output.parent / f"{args.output.stem}_{metric_column}_histogram.png"
    plot_attribution_histogram(per_unit, histogram_path, value_column=metric_column)
    print(f"Saved attribution histogram to {histogram_path}")


if __name__ == "__main__":
    main()
