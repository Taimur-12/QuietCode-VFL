"""
Generate Pareto curve and supporting bit-efficiency plots from result JSONs.
Run locally: python make_figures.py
Outputs go to figures/ directory.
"""

import json, os, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results_final")
FIG_DIR     = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

METHOD_SPECS = [
    {"name": "A_plain_vfl",      "family": "plain",     "comm_bits": 40960},
    {"name": "A_proj_vfl",       "family": "proj",      "comm_bits": 4096},
    {"name": "S_rand_sign",      "family": "rand_sign", "comm_bits": 128},
    {"name": "S_sign_quant",     "family": "sign",      "comm_bits": 128},
    {"name": "H_vq_K64",         "family": "vq",        "comm_bits": 48},
    {"name": "H_vq_K256",        "family": "vq",        "comm_bits": 64},
    {"name": "H_vq_no_kmeans",   "family": "vq",        "comm_bits": 64},
    {"name": "H_vq_M4",          "family": "vq",        "comm_bits": 32},
    {"name": "H_vq_M16",         "family": "vq",        "comm_bits": 128},
    {"name": "H_vq_commit_low",  "family": "vq",        "comm_bits": 64},
    {"name": "H_vq_commit_high", "family": "vq",        "comm_bits": 64},
]

SEEDS = [42, 43]

FAMILY_COLORS = {
    "plain":     "#444444",
    "proj":      "#888888",
    "sign":      "#1f77b4",
    "rand_sign": "#aec7e8",
    "vq":        "#d62728",
}

FAMILY_LABELS = {
    "plain":     "Plain VFL",
    "proj":      "Projection",
    "sign":      "Sign quant",
    "rand_sign": "Random sign",
    "vq":        "Product VQ",
}


def _mean_std(vals):
    arr = np.array([v for v in vals if v is not None and not np.isnan(v)])
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())


def load_results():
    rows = []
    for spec in METHOD_SPECS:
        wacc_vals, ssim_vals, lpips_vals = [], [], []
        for seed in SEEDS:
            rpath = os.path.join(RESULTS_DIR, f"result_{spec['name']}_seed{seed}.json")
            rcpath = os.path.join(RESULTS_DIR, f"recon_{spec['name']}_seed{seed}.json")
            if os.path.exists(rpath):
                with open(rpath) as f:
                    d = json.load(f)
                wacc_vals.append(d.get("best_wacc"))
            if os.path.exists(rcpath):
                with open(rcpath) as f:
                    d = json.load(f)
                if not d.get("skipped", False):
                    ssim_vals.append(d.get("ssim_mean"))
                    lpips_vals.append(d.get("lpips_mean"))

        wm, ws = _mean_std(wacc_vals)
        sm, ss = _mean_std(ssim_vals)
        lm, ls = _mean_std(lpips_vals)
        rows.append({
            "method":     spec["name"],
            "family":     spec["family"],
            "comm_bits":  spec["comm_bits"],
            "wacc_mean":  wm, "wacc_std":  ws,
            "ssim_mean":  sm, "ssim_std":  ss,
            "lpips_mean": lm, "lpips_std": ls,
        })
    return pd.DataFrame(rows)


def plot_pareto(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    seen_families = set()

    for _, r in df.iterrows():
        if any(np.isnan([r["ssim_mean"], r["wacc_mean"]])): continue
        fam = r["family"]
        color = FAMILY_COLORS.get(fam, "#000000")
        label = FAMILY_LABELS.get(fam, fam) if fam not in seen_families else None
        seen_families.add(fam)

        ms = 5 + 12 * (1 - (math.log2(r["comm_bits"]) / math.log2(40960)))
        ax.errorbar(
            r["ssim_mean"], r["wacc_mean"],
            xerr=r.get("ssim_std", 0) or 0,
            yerr=r.get("wacc_std", 0) or 0,
            fmt="o", color=color, capsize=3,
            markersize=max(6, ms), label=label,
            markeredgewidth=0.5, markeredgecolor="white",
        )
        short = r["method"].replace("H_vq_", "").replace("A_", "").replace("S_", "")
        ax.annotate(
            f"{short}\n{int(r['comm_bits'])}b",
            (r["ssim_mean"], r["wacc_mean"]),
            fontsize=7.5, xytext=(6, 4), textcoords="offset points",
        )

    ax.set_xlabel("Reconstruction SSIM  (↑ easier to reconstruct, less private)", fontsize=11)
    ax.set_ylabel("Validation WACC  (↑ better utility)", fontsize=11)
    ax.set_title("Privacy–Utility Pareto: all 11 methods on ISIC-2019", fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend(title="Method family", loc="lower left", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "pareto_wacc_vs_ssim.pdf")
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_bits_vs(df, metric, ylabel, fname, invert_y=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    seen_families = set()
    for _, r in df.iterrows():
        v = r[f"{metric}_mean"]
        if np.isnan(v): continue
        fam = r["family"]
        color = FAMILY_COLORS.get(fam, "#000000")
        label = FAMILY_LABELS.get(fam, fam) if fam not in seen_families else None
        seen_families.add(fam)
        ax.errorbar(
            r["comm_bits"], v,
            yerr=r.get(f"{metric}_std", 0) or 0,
            fmt="o", color=color, capsize=3, markersize=8, label=label,
        )
        short = r["method"].replace("H_vq_", "").replace("A_", "").replace("S_", "")
        ax.annotate(short, (r["comm_bits"], v), fontsize=8,
                    xytext=(5, 3), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Communication bits per sample (log scale)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(alpha=0.3, which="both")
    if invert_y: ax.invert_yaxis()
    ax.legend(title="Method family", loc="best", fontsize=9)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, fname)
    fig.savefig(out, dpi=150)
    fig.savefig(out.replace(".pdf", ".png"), dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    df = load_results()
    print(df[["method", "comm_bits", "wacc_mean", "wacc_std", "ssim_mean", "lpips_mean"]].to_string(index=False))
    plot_pareto(df)
    plot_bits_vs(df, "ssim",  "Reconstruction SSIM (↑ less private)", "bits_vs_ssim.pdf")
    plot_bits_vs(df, "wacc",  "Validation WACC (↑ better)",            "bits_vs_wacc.pdf")
    plot_bits_vs(df, "lpips", "LPIPS (↑ more private)",                 "bits_vs_lpips.pdf", invert_y=False)
    print(f"\nAll figures saved to {FIG_DIR}/")
