#!/usr/bin/env python
# Reanalysis of subjective explanation-quality ratings for six LLMs.
# Input workbook is read with artifact_tool; statistical calculations use NumPy/SciPy;
# output workbook is written with artifact_tool.
#
# Usage:
#   python analyze_table3.py --input data_table3.xlsx --output_dir table3_outputs
#
# Required packages:
#   artifact_tool, numpy, scipy, matplotlib
#
# Primary analysis:
# - Friedman test separately for each rater within each Level x Subject stratum.
# - Holm adjustment across the 12 omnibus tests separately for each rater.
# - Pairwise two-sided Wilcoxon signed-rank tests, with Holm correction across
#   the 15 model pairs within each Level x Subject x Rater stratum.
# - Kendall's W = Friedman Q / [n * (k - 1)].
# - Score distribution (1-5) and low-score proportion (scores 1-2).
# - Inter-rater ICC(A,1) absolute-agreement point estimate and item-bootstrap CI
#   by professional level and model (provided as an additional reliability analysis).

import argparse
import itertools
import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

from artifact_tool import Blob, SpreadsheetFile, Workbook


MODELS = [
    "Kimi K2",
    "DeepSeek R1",
    "Qwen3-Max",
    "Doubao 1.6",
    "gemini flash 2.5",
    "chatgpt 4o-mini",
]
RATERS = ["Rater 1", "Rater 2"]
RATER_PREFIX = {"Rater 1": "Rater1_", "Rater 2": "Rater2_"}

LEVELS = [
    "Junior Radiographer",
    "Radiographer",
    "Radiographer-in-charge",
]
SUBJECTS = [
    "Basic Knowledge",
    "Relevant Professional Knowledge",
    "Professional Knowledge",
    "Professional Practice Skills",
]

EXPECTED_COLS = (
    ["Question_ID", "Level", "Subject"]
    + [RATER_PREFIX["Rater 1"] + m for m in MODELS]
    + [RATER_PREFIX["Rater 2"] + m for m in MODELS]
)


def holm_adjust(p_values):
    """Holm step-down adjusted p-values; preserves order."""
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan, dtype=float)
    good = np.isfinite(p)
    if not np.any(good):
        return out.tolist()
    pv = p[good]
    m = len(pv)
    order = np.argsort(pv)
    sorted_p = pv[order]
    adjusted_sorted = np.maximum.accumulate((m - np.arange(m)) * sorted_p)
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)
    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    out[good] = adjusted
    return out.tolist()


def p_text(p):
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def significance(p):
    if p is None or not np.isfinite(p):
        return "NA"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def median_iqr_string(x):
    x = np.asarray(x, dtype=float)
    q1, med, q3 = np.percentile(x, [25, 50, 75], method="linear")
    return f"{med:.2f} ({q1:.2f}, {q3:.2f})"


def icc_a1_absolute_agreement(x):
    """
    Two-way absolute-agreement single-measure ICC(A,1), commonly denoted ICC(2,1).
    Rows are subjects/items, columns are the two raters.
    """
    x = np.asarray(x, dtype=float)
    n, k = x.shape
    if n < 2 or k < 2:
        return np.nan
    grand = np.mean(x)
    row_means = np.mean(x, axis=1)
    col_means = np.mean(x, axis=0)
    ms_subject = k * np.sum((row_means - grand) ** 2) / (n - 1)
    ms_rater = n * np.sum((col_means - grand) ** 2) / (k - 1)
    residual = x - row_means[:, None] - col_means[None, :] + grand
    ms_error = np.sum(residual ** 2) / ((n - 1) * (k - 1))
    denom = ms_subject + (k - 1) * ms_error + k * (ms_rater - ms_error) / n
    if denom == 0:
        return np.nan
    return (ms_subject - ms_error) / denom


def bootstrap_icc_ci(x, n_boot=1000, seed=20261105):
    x = np.asarray(x, dtype=float)
    point = icc_a1_absolute_agreement(x)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    vals = []
    if n < 2:
        return point, np.nan, np.nan
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        value = icc_a1_absolute_agreement(x[idx, :])
        if np.isfinite(value):
            vals.append(value)
    if len(vals) < max(100, int(0.5 * n_boot)):
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)



def weighted_cohens_kappa(a, b, weight_type="quadratic"):
    """Weighted Cohen's kappa for paired ordinal ratings (categories 1-5)."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    categories = np.arange(1, 6)
    cm = np.zeros((5, 5), dtype=float)
    for av, bv in zip(a, b):
        cm[av - 1, bv - 1] += 1.0
    n = cm.sum()
    if n == 0:
        return np.nan
    observed = cm / n
    row_marginal = observed.sum(axis=1)
    col_marginal = observed.sum(axis=0)
    expected = np.outer(row_marginal, col_marginal)
    distance = np.abs(categories[:, None] - categories[None, :]) / 4.0
    if weight_type == "linear":
        weights = 1.0 - distance
    else:
        weights = 1.0 - distance ** 2
    po = float(np.sum(weights * observed))
    pe = float(np.sum(weights * expected))
    if np.isclose(1.0 - pe, 0.0):
        return np.nan
    return (po - pe) / (1.0 - pe)


def bootstrap_weighted_kappa_ci(a, b, weight_type="quadratic", n_boot=1000, seed=20261105):
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    point = weighted_cohens_kappa(a, b, weight_type=weight_type)
    n = len(a)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        val = weighted_cohens_kappa(a[idx], b[idx], weight_type=weight_type)
        if np.isfinite(val):
            vals.append(val)
    if len(vals) < max(100, int(0.5 * n_boot)):
        return point, np.nan, np.nan
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi)

def load_input(input_path):
    source_wb = SpreadsheetFile.import_xlsx(Blob.load(str(input_path)))
    sheet_info = source_wb.inspect({"kind": "sheet", "include": "id,name"}).ndjson
    first_sheet = json.loads(sheet_info.splitlines()[0])["name"]
    src_sheet = source_wb.worksheets.get_item(first_sheet)
    # The submitted ratings sheet has 15 columns and about 2,000 records.
    raw = src_sheet.get_range("A1:O10000").values
    raw = [r for r in raw if r and any(v is not None for v in r)]
    headers = raw[0]
    data = raw[1:]

    missing = [c for c in EXPECTED_COLS if c not in headers]
    if missing:
        raise ValueError(
            "The input workbook does not match the expected column names. "
            f"Missing columns: {missing}"
        )

    col_idx = {name: headers.index(name) for name in EXPECTED_COLS}
    records = []
    for row_number, row in enumerate(data, start=2):
        rec = {name: row[col_idx[name]] for name in EXPECTED_COLS}
        if rec["Question_ID"] is None:
            raise ValueError(f"Missing Question_ID at Excel row {row_number}.")
        if rec["Level"] not in LEVELS:
            raise ValueError(f"Unexpected Level value at Excel row {row_number}: {rec['Level']}")
        if rec["Subject"] not in SUBJECTS:
            raise ValueError(f"Unexpected Subject value at Excel row {row_number}: {rec['Subject']}")
        for col in EXPECTED_COLS[3:]:
            value = rec[col]
            if value is None:
                raise ValueError(f"Missing rating at Excel row {row_number}, column {col}.")
            if value not in [1, 2, 3, 4, 5]:
                raise ValueError(f"Rating must be 1-5; found {value} at Excel row {row_number}, column {col}.")
        records.append(rec)

    ids = [r["Question_ID"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError(
            "Question_ID is not unique in the current file. Please add an item-level unique ID "
            "or resolve duplicate IDs before matched analysis."
        )
    return records, headers, first_sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_table3.xlsx")
    parser.add_argument("--output_dir", default="table3_outputs")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records, headers, sheet_name = load_input(input_path)
    n_items = len(records)

    # QC summaries
    level_counts = {lev: sum(r["Level"] == lev for r in records) for lev in LEVELS}
    stratum_counts = {
        (lev, subj): sum(r["Level"] == lev and r["Subject"] == subj for r in records)
        for lev in LEVELS for subj in SUBJECTS
    }
    all_scores = [
        r[c] for r in records for c in EXPECTED_COLS[3:]
    ]
    score_counts = {score: sum(v == score for v in all_scores) for score in range(1, 6)}
    missing_scores = sum(v is None for r in records for v in [r[c] for c in EXPECTED_COLS[3:]])

    # Build Table 3: per rater × level × subject.
    main_rows = []
    omnibus_rows = []
    pairwise_rows = []
    distribution_rows = []
    low_score_rows = []

    # Cache paired matrices by rater/stratum for plotting and statistical tests.
    cache = {}

    for rater in RATERS:
        prefix = RATER_PREFIX[rater]
        current_omnibus_indices = []

        for lev in LEVELS:
            for subj in SUBJECTS:
                sub = [r for r in records if r["Level"] == lev and r["Subject"] == subj]
                n = len(sub)
                if n == 0:
                    continue
                matrix = np.column_stack([
                    np.asarray([r[prefix + m] for r in sub], dtype=float)
                    for m in MODELS
                ])
                cache[(rater, lev, subj)] = matrix

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    friedman = stats.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
                q_stat = float(friedman.statistic) if np.isfinite(friedman.statistic) else np.nan
                p_raw = float(friedman.pvalue) if np.isfinite(friedman.pvalue) else np.nan
                kendall_w = q_stat / (n * (len(MODELS) - 1)) if np.isfinite(q_stat) else np.nan

                med_iqr = {
                    m: median_iqr_string(matrix[:, j])
                    for j, m in enumerate(MODELS)
                }
                omnibus_rows.append({
                    "Rater": rater,
                    "Level": lev,
                    "Subject": subj,
                    "n_questions": n,
                    "Friedman_Q": q_stat,
                    "df": len(MODELS) - 1,
                    "P_raw": p_raw,
                    "P_Holm_within_rater_12_strata": np.nan,
                    "Kendall_W": kendall_w,
                    "Significance_raw": significance(p_raw),
                })
                current_omnibus_indices.append(len(omnibus_rows) - 1)

                main_row = {
                    "Rater": rater,
                    "Level": lev,
                    "Subject": subj,
                    "Questions_n": n,
                }
                for m in MODELS:
                    main_row[m + "_Median_IQR"] = med_iqr[m]
                main_row.update({
                    "Friedman_Q": q_stat,
                    "df": len(MODELS) - 1,
                    "P_raw": p_raw,
                    "P_Holm": np.nan,
                    "Kendall_W": kendall_w,
                })
                main_rows.append(main_row)

                # Distribution and low-score rate (scores 1-2) per model.
                for j, model in enumerate(MODELS):
                    vals = matrix[:, j].astype(int)
                    low_n = int(np.sum(vals <= 2))
                    low_score_rows.append({
                        "Rater": rater,
                        "Level": lev,
                        "Subject": subj,
                        "Model": model,
                        "n_questions": n,
                        "Score_1_n": int(np.sum(vals == 1)),
                        "Score_2_n": int(np.sum(vals == 2)),
                        "Low_score_1_2_n": low_n,
                        "Low_score_1_2_pct": 100.0 * low_n / n,
                    })
                    for score in range(1, 6):
                        count = int(np.sum(vals == score))
                        distribution_rows.append({
                            "Rater": rater,
                            "Level": lev,
                            "Subject": subj,
                            "Model": model,
                            "Score": score,
                            "n": count,
                            "Percent": 100.0 * count / n,
                        })

                # Pairwise Wilcoxon signed-rank tests, with Holm within this stratum.
                pending = []
                for i, j in itertools.combinations(range(len(MODELS)), 2):
                    x = matrix[:, i]
                    y = matrix[:, j]
                    diff = x - y
                    nonzero = int(np.sum(diff != 0))
                    median_diff = float(np.median(diff))
                    if nonzero == 0:
                        stat_value, p_pair = 0.0, 1.0
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            test = stats.wilcoxon(
                                x, y,
                                zero_method="wilcox",
                                correction=True,
                                alternative="two-sided",
                                method="approx",
                            )
                        stat_value = float(test.statistic)
                        p_pair = float(test.pvalue)
                    pending.append({
                        "Rater": rater,
                        "Level": lev,
                        "Subject": subj,
                        "n_questions": n,
                        "Model_1": MODELS[i],
                        "Model_2": MODELS[j],
                        "Median_paired_difference_Model1_minus_Model2": median_diff,
                        "Nonzero_pairs": nonzero,
                        "Wilcoxon_W": stat_value,
                        "P_raw": p_pair,
                    })
                adjusted = holm_adjust([x["P_raw"] for x in pending])
                for entry, p_adj in zip(pending, adjusted):
                    entry["P_Holm_within_stratum_15_pairs"] = p_adj
                    entry["Significance_Holm"] = significance(p_adj)
                    pairwise_rows.append(entry)

        # Holm correction across the 12 omnibus strata separately for each rater.
        indices = current_omnibus_indices
        adj = holm_adjust([omnibus_rows[i]["P_raw"] for i in indices])
        for i, p_adj in zip(indices, adj):
            omnibus_rows[i]["P_Holm_within_rater_12_strata"] = p_adj
            omnibus_rows[i]["Significance_Holm"] = significance(p_adj)

    # Place corrected omnibus values back into Table 3 main rows.
    omnibus_lookup = {
        (r["Rater"], r["Level"], r["Subject"]): r
        for r in omnibus_rows
    }
    for row in main_rows:
        o = omnibus_lookup[(row["Rater"], row["Level"], row["Subject"])]
        row["P_Holm"] = o["P_Holm_within_rater_12_strata"]
        row["Significance_Holm"] = o["Significance_Holm"]

    # Mark whether pairwise post-hoc comparisons follow a significant Holm-adjusted omnibus test.
    for row in pairwise_rows:
        o = omnibus_lookup[(row["Rater"], row["Level"], row["Subject"])]
        row["Omnibus_Holm_significant"] = bool(
            np.isfinite(o["P_Holm_within_rater_12_strata"])
            and o["P_Holm_within_rater_12_strata"] < 0.05
        )

    # Overall score distribution by rater/model, pooling all levels and subjects.
    overall_distribution = []
    for rater in RATERS:
        prefix = RATER_PREFIX[rater]
        for model in MODELS:
            vals = np.asarray([r[prefix + model] for r in records], dtype=int)
            n = len(vals)
            low_n = int(np.sum(vals <= 2))
            for score in range(1, 6):
                cnt = int(np.sum(vals == score))
                overall_distribution.append({
                    "Rater": rater,
                    "Model": model,
                    "Score": score,
                    "n": cnt,
                    "Percent": 100.0 * cnt / n,
                    "Total_n": n,
                    "Low_score_1_2_n": low_n,
                    "Low_score_1_2_pct": 100.0 * low_n / n,
                })

    # Additional inter-rater reliability, matching the structure of the old Table 3:
    # one ICC(A,1) per professional level and model, pooled across subjects.
    icc_rows = []
    for lev in LEVELS:
        sub = [r for r in records if r["Level"] == lev]
        for model in MODELS:
            x = np.column_stack([
                np.asarray([r[RATER_PREFIX["Rater 1"] + model] for r in sub], dtype=float),
                np.asarray([r[RATER_PREFIX["Rater 2"] + model] for r in sub], dtype=float),
            ])
            seed = 20261105 + LEVELS.index(lev) * 100 + MODELS.index(model)
            point, lo, hi = bootstrap_icc_ci(
                x,
                n_boot=args.bootstrap,
                seed=seed,
            )
            k_linear, kl_lo, kl_hi = bootstrap_weighted_kappa_ci(
                x[:, 0], x[:, 1], weight_type="linear",
                n_boot=args.bootstrap, seed=seed + 1000
            )
            k_quad, kq_lo, kq_hi = bootstrap_weighted_kappa_ci(
                x[:, 0], x[:, 1], weight_type="quadratic",
                n_boot=args.bootstrap, seed=seed + 2000
            )
            exact_agreement = float(np.mean(x[:, 0] == x[:, 1]) * 100.0)
            icc_rows.append({
                "Level": lev,
                "Model": model,
                "n_questions": len(sub),
                "ICC_A1_absolute_agreement": point,
                "ICC_bootstrap_95CI_low": lo,
                "ICC_bootstrap_95CI_high": hi,
                "Linear_weighted_kappa": k_linear,
                "Linear_kappa_bootstrap_95CI_low": kl_lo,
                "Linear_kappa_bootstrap_95CI_high": kl_hi,
                "Quadratic_weighted_kappa": k_quad,
                "Quadratic_kappa_bootstrap_95CI_low": kq_lo,
                "Quadratic_kappa_bootstrap_95CI_high": kq_hi,
                "Exact_agreement_pct": exact_agreement,
                "Bootstrap_replicates_requested": args.bootstrap,
            })

    # QC table, including differences from the previous Table 3 counts visible in the supplied PDF.
    previous_counts = {
        ("Junior Radiographer", "Basic Knowledge"): 414,
        ("Junior Radiographer", "Relevant Professional Knowledge"): 109,
        ("Junior Radiographer", "Professional Knowledge"): 52,
        ("Junior Radiographer", "Professional Practice Skills"): 72,
        ("Radiographer", "Basic Knowledge"): 344,
        ("Radiographer", "Relevant Professional Knowledge"): 177,
        ("Radiographer", "Professional Knowledge"): 46,
        ("Radiographer", "Professional Practice Skills"): 176,
        ("Radiographer-in-charge", "Basic Knowledge"): 361,
        ("Radiographer-in-charge", "Relevant Professional Knowledge"): 140,
        ("Radiographer-in-charge", "Professional Knowledge"): 44,
        ("Radiographer-in-charge", "Professional Practice Skills"): 55,
    }
    qc_rows = [
        {"QC_item": "Number of questions", "Value": n_items, "Note": "Current uploaded workbook"},
        {"QC_item": "Unique Question_ID", "Value": len(set(r["Question_ID"] for r in records)), "Note": "Must equal number of rows"},
        {"QC_item": "Missing ratings", "Value": missing_scores, "Note": "Across 12 rater-model score columns"},
        {"QC_item": "Minimum score", "Value": int(min(all_scores)), "Note": "Expected 1"},
        {"QC_item": "Maximum score", "Value": int(max(all_scores)), "Note": "Expected 5"},
    ]
    for s in range(1, 6):
        qc_rows.append({
            "QC_item": f"All ratings: score {s}",
            "Value": score_counts[s],
            "Note": f"{100.0*score_counts[s]/len(all_scores):.2f}% of all {len(all_scores)} ratings",
        })
    for lev in LEVELS:
        qc_rows.append({
            "QC_item": f"Questions — {lev}",
            "Value": level_counts[lev],
            "Note": "Current workbook",
        })
    for lev in LEVELS:
        for subj in SUBJECTS:
            old_n = previous_counts[(lev, subj)]
            new_n = stratum_counts[(lev, subj)]
            if new_n != old_n:
                qc_rows.append({
                    "QC_item": f"Count difference vs old Table 3 — {lev} / {subj}",
                    "Value": new_n,
                    "Note": f"Old Table 3 n={old_n}; current workbook n={new_n}; difference={new_n-old_n}",
                })

    # --- Figures ---
    # Figure 2: 6-panel 100% stacked horizontal bars (Rater x Level), all four domains pooled.
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.0), sharex=True)
    score_labels = [f"Score {v}" for v in range(1, 6)]
    for row_i, lev in enumerate(LEVELS):
        for col_i, rater in enumerate(RATERS):
            ax = axes[row_i, col_i]
            prefix = RATER_PREFIX[rater]
            sub = [r for r in records if r["Level"] == lev]
            y_positions = np.arange(len(MODELS))
            left = np.zeros(len(MODELS), dtype=float)
            for score in range(1, 6):
                widths = []
                for model in MODELS:
                    vals = np.asarray([r[prefix + model] for r in sub], dtype=int)
                    widths.append(100.0 * np.mean(vals == score))
                ax.barh(
                    y_positions,
                    widths,
                    left=left,
                    height=0.72,
                    label=score_labels[score - 1] if row_i == 0 and col_i == 0 else None,
                )
                left += np.asarray(widths)
            ax.set_yticks(y_positions)
            ax.set_yticklabels(MODELS, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlim(0, 100)
            ax.set_title(f"{lev} — {rater}", fontsize=10, weight="bold")
            panel_letter = chr(65 + row_i * 2 + col_i)
            ax.text(-0.12, 1.08, f"({panel_letter})", transform=ax.transAxes,
                    fontsize=11, fontweight="bold", va="bottom", ha="left")
            ax.grid(axis="x", alpha=0.25)
            ax.set_axisbelow(True)
            if row_i == 2:
                ax.set_xlabel("Rating distribution (%)")
            if col_i == 0:
                ax.set_ylabel("Model")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if not handles:
        handles = axes[0, 0].containers
        labels = score_labels
    fig.legend(
        handles[:5], score_labels,
        loc="lower center", ncol=5, frameon=False,
        bbox_to_anchor=(0.5, 0.01), title="Likert score"
    )
    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(output_dir / "Figure2_Subjective_Score_Distribution.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "Figure2_Subjective_Score_Distribution.pdf", bbox_inches="tight")
    plt.close(fig)

    # Figure S2: low-score (1-2) proportion heatmaps, separate panels for raters.
    fig, axes = plt.subplots(1, 2, figsize=(17, 7), sharey=True)
    xlabels = [f"{lev.replace(' Radiographer',' Rad.').replace('Radiographer-in-charge','In-charge')}\n{subj.replace(' Knowledge','').replace('Professional Practice Skills','Practice Skills')}"
               for lev in LEVELS for subj in SUBJECTS]
    for ax_i, rater in enumerate(RATERS):
        ax = axes[ax_i]
        heat = np.zeros((len(MODELS), len(LEVELS) * len(SUBJECTS)), dtype=float)
        for mi, model in enumerate(MODELS):
            ci = 0
            for lev in LEVELS:
                for subj in SUBJECTS:
                    entry = next(x for x in low_score_rows
                                 if x["Rater"] == rater and x["Level"] == lev
                                 and x["Subject"] == subj and x["Model"] == model)
                    heat[mi, ci] = entry["Low_score_1_2_pct"]
                    ci += 1
        image = ax.imshow(heat, aspect="auto", interpolation="nearest")
        panel_letter = "A" if ax_i == 0 else "B"
        ax.set_title(f"({panel_letter}) {rater}", fontsize=11, weight="bold")
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation=50, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(MODELS)))
        ax.set_yticklabels(MODELS, fontsize=8)
        ax.set_xlabel("Professional level / knowledge domain")
        if ax_i == 0:
            ax.set_ylabel("Model")
        for i in range(len(MODELS)):
            for j in range(len(xlabels)):
                ax.text(j, i, f"{heat[i,j]:.1f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02, label="Low ratings (scores 1–2), %")
    fig.tight_layout(rect=[0, 0, 1, 1])
    fig.savefig(output_dir / "FigureS2_Low_Score_Heatmap.png", dpi=600, bbox_inches="tight")
    fig.savefig(output_dir / "FigureS2_Low_Score_Heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- Output workbook (artifact_tool) ---
    wb_out = Workbook.create()

    def add_sheet(name, headers_out, rows_out, widths=None):
        sh = wb_out.worksheets.add(name)
        matrix = [headers_out]
        for row in rows_out:
            matrix.append([row.get(h) for h in headers_out])
        sh.get_range_by_indexes(0, 0, len(matrix), len(headers_out)).values = matrix
        header_rng = sh.get_range_by_indexes(0, 0, 1, len(headers_out))
        header_rng.format = {
            "fill": "#1F4E78",
            "font": {"bold": True, "color": "#FFFFFF"},
            "horizontal_alignment": "center",
            "vertical_alignment": "center",
            "wrap_text": True,
            "row_height": 32,
        }
        sh.freeze_panes.freeze_rows(1)
        sh.get_range_by_indexes(1, 0, max(1, len(rows_out)), len(headers_out)).format.wrap_text = True
        if widths:
            for col_idx, width in enumerate(widths):
                sh.get_range_by_indexes(0, col_idx, max(1, len(matrix)), 1).format.column_width = width
        else:
            sh.get_range_by_indexes(0, 0, len(matrix), len(headers_out)).format.autofit_columns()
        return sh

    main_headers = (
        ["Rater", "Level", "Subject", "Questions_n"]
        + [m + "_Median_IQR" for m in MODELS]
        + ["Friedman_Q", "df", "P_raw", "P_Holm", "Kendall_W", "Significance_Holm"]
    )
    add_sheet(
        "Table3_Main",
        main_headers,
        main_rows,
        widths=[13, 25, 34, 12] + [22]*6 + [13, 8, 12, 12, 12, 15],
    )

    omnibus_headers = [
        "Rater", "Level", "Subject", "n_questions", "Friedman_Q", "df",
        "P_raw", "P_Holm_within_rater_12_strata", "Kendall_W",
        "Significance_raw", "Significance_Holm"
    ]
    add_sheet("Friedman_Omnibus", omnibus_headers, omnibus_rows,
              widths=[13, 25, 34, 12, 13, 8, 12, 22, 12, 15, 15])

    pair_headers = [
        "Rater", "Level", "Subject", "n_questions", "Model_1", "Model_2",
        "Median_paired_difference_Model1_minus_Model2", "Nonzero_pairs",
        "Wilcoxon_W", "P_raw", "P_Holm_within_stratum_15_pairs",
        "Significance_Holm", "Omnibus_Holm_significant"
    ]
    add_sheet("Wilcoxon_Pairwise", pair_headers, pairwise_rows,
              widths=[13, 25, 34, 12, 20, 20, 28, 14, 13, 12, 28, 16])

    dist_headers = ["Rater", "Level", "Subject", "Model", "Score", "n", "Percent"]
    add_sheet("Score_Distribution", dist_headers, distribution_rows,
              widths=[13, 25, 34, 20, 10, 10, 12])

    low_headers = [
        "Rater", "Level", "Subject", "Model", "n_questions",
        "Score_1_n", "Score_2_n", "Low_score_1_2_n", "Low_score_1_2_pct"
    ]
    add_sheet("Low_Score_Rate", low_headers, low_score_rows,
              widths=[13, 25, 34, 20, 12, 12, 12, 18, 20])

    overall_headers = [
        "Rater", "Model", "Score", "n", "Percent", "Total_n",
        "Low_score_1_2_n", "Low_score_1_2_pct"
    ]
    add_sheet("Overall_Distribution", overall_headers, overall_distribution,
              widths=[13, 22, 10, 10, 12, 12, 18, 20])

    icc_headers = [
        "Level", "Model", "n_questions",
        "ICC_A1_absolute_agreement", "ICC_bootstrap_95CI_low", "ICC_bootstrap_95CI_high",
        "Linear_weighted_kappa", "Linear_kappa_bootstrap_95CI_low", "Linear_kappa_bootstrap_95CI_high",
        "Quadratic_weighted_kappa", "Quadratic_kappa_bootstrap_95CI_low", "Quadratic_kappa_bootstrap_95CI_high",
        "Exact_agreement_pct", "Bootstrap_replicates_requested"
    ]
    add_sheet("Interrater_Reliability", icc_headers, icc_rows,
              widths=[25, 22, 12, 25, 20, 20, 22, 26, 26, 24, 28, 28, 20, 26])

    qc_headers = ["QC_item", "Value", "Note"]
    add_sheet("Data_QC", qc_headers, qc_rows, widths=[54, 16, 72])

    # Readme in workbook.
    readme_rows = [
        {"Item": "Input", "Description": str(input_path)},
        {"Item": "Question count", "Description": str(n_items)},
        {"Item": "Primary test", "Description": "Friedman test run separately for Rater 1 and Rater 2 in each Level × Subject stratum."},
        {"Item": "Omnibus Holm correction", "Description": "12 Friedman P values corrected within each rater."},
        {"Item": "Pairwise test", "Description": "Two-sided paired Wilcoxon signed-rank test; zero differences omitted by zero_method='wilcox'; continuity correction; normal approximation."},
        {"Item": "Pairwise Holm correction", "Description": "15 model-pair P values corrected within each Rater × Level × Subject stratum."},
        {"Item": "Effect size", "Description": "Kendall's W = Friedman Q / [n × (6 − 1)]."},
        {"Item": "Low-score definition", "Description": "Low score defined a priori for this analysis as rating 1 or 2 (≤2)."},
        {"Item": "Inter-rater reliability", "Description": "Additional analysis: ICC(A,1) absolute-agreement single-measure with item-bootstrap 95% CI, plus linear and quadratic weighted Cohen kappa with bootstrap 95% CI; all by professional level and model."},
        {"Item": "Model labels", "Description": "Exact labels from current Excel used: gemini flash 2.5 and chatgpt 4o-mini. Verify API model versions before manuscript submission."},
        {"Item": "Old Table 3 difference", "Description": "New workbook contains 1,994 questions; prior Table 3 showed 1,990. See Data_QC for subgroup discrepancies."},
    ]
    add_sheet("Readme", ["Item", "Description"], readme_rows, widths=[28, 120])

    output_xlsx = output_dir / "Table3_Statistical_Results.xlsx"
    SpreadsheetFile.export_xlsx(wb_out).save(str(output_xlsx))

    # CSV outputs for reproducibility and direct import into other software.
    import csv
    def write_csv(file_name, headers_out, rows_out):
        with (output_dir / file_name).open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=headers_out)
            writer.writeheader()
            writer.writerows(rows_out)

    write_csv("Table3_Main.csv", main_headers, main_rows)
    write_csv("Friedman_Omnibus.csv", omnibus_headers, omnibus_rows)
    write_csv("Wilcoxon_Pairwise.csv", pair_headers, pairwise_rows)
    write_csv("Score_Distribution.csv", dist_headers, distribution_rows)
    write_csv("Low_Score_Rate.csv", low_headers, low_score_rows)
    write_csv("Interrater_Reliability.csv", icc_headers, icc_rows)
    write_csv("Data_QC.csv", qc_headers, qc_rows)

    # Console summary
    print(f"Analysis completed: {n_items} questions, {len(main_rows)} Table 3 rows.")
    print(f"All rating values observed: {sorted(set(all_scores))}; missing ratings={missing_scores}.")
    print("\nFriedman omnibus results (Holm-adjusted across 12 strata within each rater):")
    for r in omnibus_rows:
        print(
            f"{r['Rater']} | {r['Level']} | {r['Subject']} | n={r['n_questions']} | "
            f"Q={r['Friedman_Q']:.3f} | P={p_text(r['P_raw'])} | "
            f"Holm P={p_text(r['P_Holm_within_rater_12_strata'])} | "
            f"Kendall W={r['Kendall_W']:.3f}"
        )
    sig_pairs = sum(
        1 for r in pairwise_rows
        if np.isfinite(r["P_Holm_within_stratum_15_pairs"])
        and r["P_Holm_within_stratum_15_pairs"] < 0.05
    )
    print(f"\nSignificant pairwise Wilcoxon comparisons after within-stratum Holm: {sig_pairs}/{len(pairwise_rows)}.")
    print(f"Output workbook: {output_xlsx}")
    print(f"Figures: Figure2_Subjective_Score_Distribution.png/.pdf; FigureS2_Low_Score_Heatmap.png/.pdf")


if __name__ == "__main__":
    main()
