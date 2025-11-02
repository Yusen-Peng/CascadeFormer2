import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def visualize_knn_maha_scatter(
    incidents_df: pd.DataFrame,
    outpath: str = "knn_maha_scatter.png",
    knn_thresh: float | None = None,
    maha_thresh: float | None = None,
    log_maha: bool = False,
    s: int = 12,
    alpha: float = 0.6,
) -> str:
    """
    Save a 2D scatter plot of KNN distance vs Mahalanobis distance for all incidents.

    Args:
        incidents_df: DataFrame with columns ["knn_dist", "mahalanobis", "gt_label"].
        outpath: where to save the figure.
        knn_thresh: optional vertical guide line at this KNN distance.
        maha_thresh: optional horizontal guide line at this Mahalanobis distance.
        log_maha: if True, plot log10(mahalanobis) on the Y axis to stabilize scale.
        s: point size.
        alpha: point transparency.

    Returns:
        The saved figure path.
    """
    assert {"knn_dist", "mahalanobis", "gt_label"}.issubset(incidents_df.columns), \
        "incidents_df must have columns: knn_dist, mahalanobis, gt_label"

    x = incidents_df["knn_dist"].values
    yraw = incidents_df["mahalanobis"].values
    y = (np.log10(np.clip(yraw, 1e-12, None)) if log_maha else yraw)

    # Split by ground-truth label for quick visual separation
    is_abn = incidents_df["gt_label"].astype(str).str.lower().eq("abnormal")
    is_nor = ~is_abn

    plt.figure(figsize=(6, 5), dpi=160)
    # Keep styles minimal and reproducible
    plt.scatter(x[is_nor], y[is_nor], s=s, alpha=alpha, label="normal", marker="o")
    plt.scatter(x[is_abn], y[is_abn], s=s, alpha=alpha, label="abnormal", marker="x")

    # Optional threshold guide lines (useful for overlaying learned params later)
    if knn_thresh is not None:
        plt.axvline(knn_thresh, linestyle="--", linewidth=1)
    if maha_thresh is not None:
        plt.axhline(np.log10(maha_thresh) if (log_maha and maha_thresh is not None) else maha_thresh,
                    linestyle="--", linewidth=1)

    plt.xlabel("KNN distance")
    plt.ylabel("log10(Mahalanobis distance)" if log_maha else "Mahalanobis distance")
    plt.title("Incidents: KNN vs. Mahalanobis")
    plt.legend()
    plt.grid(True, linewidth=0.4, alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()
    return outpath