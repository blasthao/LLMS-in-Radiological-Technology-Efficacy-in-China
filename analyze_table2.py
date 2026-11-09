#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Six-LLM paired accuracy analysis for Table 2
Input:
    data_table2.xlsx
Required columns:
    ID, Question_ID, Level, Subject, Question_Type,
    Kimi K2, DeepSeek R1, Qwen3-Max, Doubao 1.6,
    gemini flash 2.5, chatgpt 4o-mini

Main analyses:
1) Binomial mixed-effects logistic model with random intercept for unique item ID
   (statsmodels BinomialBayesMixedGLM; mean-field variational Bayes).
2) Cochran's Q test for six paired binary outcomes within each Level x Subject stratum.
3) Exact McNemar tests for 15 pairwise model comparisons within each stratum,
   with Holm correction within each stratum.
4) A-only sensitivity analysis.
Outputs:
    Table2_Statistical_Results.xlsx
    Figure_Table2_Accuracy_DotWhisker.png / .pdf
    FigureS1_GLMM_OR_Forest.png / .pdf

Important:
- ID is used as the unique question identifier/random-intercept grouping variable.
  Question_ID is repeated across domains/types in the supplied file and must NOT be
  used as the random-effect ID on its own.
- The supplied file does not contain a shared-stem identifier for B-type items.
  Therefore, shared-stem clustering cannot be explicitly modeled. An A-only
  sensitivity analysis is provided.
- GLMM intervals below are approximate 95% posterior credible intervals from
  statsmodels' variational-Bayes BinomialBayesMixedGLM, not conventional
  maximum-likelihood GLMM confidence intervals.
"""

from __future__ import annotations

import argparse
import itertools
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM


MODELS = [
    "Kimi K2",
    "DeepSeek R1",
    "Qwen3-Max",
    "Doubao 1.6",
    "gemini flash 2.5",
    "chatgpt 4o-mini",
]

LEVEL_ORDER = [
    "Junior Radiographer",
    "Radiographer",
    "Radiographer-in-charge",
]

SUBJECT_ORDER = [
    "Basic Knowledge",
    "Relevant Professional Knowledge",
    "Professional Knowledge",
    "Professional Practice Skills",
]

LEVEL_SHORT = {
    "Junior Radiographer": "Junior",
    "Radiographer": "Radiographer",
    "Radiographer-in-charge": "In-charge",
}


def p_mark(p):
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def load_data(input_file: Path) -> pd.DataFrame:
    df = pd.read_excel(input_file, sheet_name=0)
    required = ["ID", "Question_ID", "Level", "Subject", "Question_Type"] + MODELS
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        raise ValueError(
            "Input file is missing required columns: " + ", ".join(missing_cols)
        )

    df = df[required].copy()

    # Normalize text fields while preserving the labels as supplied.
    for c in ["Level", "Subject", "Question_Type"]:
        df[c] = df[c].astype(str).str.strip()

    # IDs must uniquely identify rows/questions in this workbook.
    if df["ID"].isna().any():
        raise ValueError("ID contains missing values.")
    if df["ID"].duplicated().any():
        duplicated = df.loc[df["ID"].duplicated(keep=False), "ID"].head(10).tolist()
        raise ValueError(
            "ID is not unique. It is the item-level random-effect identifier. "
            f"Examples of duplicated IDs: {duplicated}"
        )

    # Verify model outcomes are binary and complete.
    for m in MODELS:
        df[m] = pd.to_numeric(df[m], errors="coerce")
        bad = df[m].notna() & ~df[m].isin([0, 1])
        if bad.any():
            raise ValueError(f"{m} contains values other than 0/1.")
    missing_outcomes = int(df[MODELS].isna().sum().sum())
    if missing_outcomes:
        raise ValueError(
            f"There are {missing_outcomes} missing model outcomes. "
            "Resolve or explicitly code missing responses before running paired tests."
        )
    for m in MODELS:
        df[m] = df[m].astype(int)

    # Categories ordered for stable tables/plots and fixed-effect reference levels.
    unknown_levels = sorted(set(df["Level"]) - set(LEVEL_ORDER))
    unknown_subjects = sorted(set(df["Subject"]) - set(SUBJECT_ORDER))
    if unknown_levels or unknown_subjects:
        raise ValueError(
            f"Unexpected category labels. Levels={unknown_levels}; Subjects={unknown_subjects}"
        )

    return df


def add_accuracy_fields(g: pd.DataFrame, model: str) -> dict:
    n = len(g)
    correct = int(g[model].sum())
    acc = correct / n if n else np.nan
    lo, hi = proportion_confint(correct, n, alpha=0.05, method="wilson")
    return {
        "n": n,
        "correct": correct,
        "accuracy": acc,
        "accuracy_pct": acc * 100,
        "wilson_low_pct": lo * 100,
        "wilson_high_pct": hi * 100,
    }


def compute_cochran_table(df: pd.DataFrame, type_filter=None):
    source = df if type_filter is None else df[df["Question_Type"].isin(type_filter)]
    records = []
    for level in LEVEL_ORDER:
        for subject in SUBJECT_ORDER:
            g = source[(source["Level"] == level) & (source["Subject"] == subject)]
            if len(g) == 0:
                continue
            qres = cochrans_q(g[MODELS].to_numpy(dtype=int), return_object=True)
            row = {
                "Level": level,
                "Subject": subject,
                "n_questions": int(len(g)),
                "Cochran_Q": float(qres.statistic),
                "P_value": float(qres.pvalue),
                "All6_incorrect_n": int((g[MODELS].eq(0).all(axis=1)).sum()),
                "All6_incorrect_pct": float((g[MODELS].eq(0).all(axis=1)).mean() * 100),
            }
            for m in MODELS:
                z = add_accuracy_fields(g, m)
                row[f"{m}_correct_n_pct"] = f'{z["correct"]} ({z["accuracy_pct"]:.1f})'
            records.append(row)

    out = pd.DataFrame(records)
    if not out.empty:
        out["Holm_adjusted_P"] = multipletests(out["P_value"].to_numpy(), method="holm")[1]
        out["Significance"] = out["Holm_adjusted_P"].map(p_mark)
    return out


def compute_mcnemar(df: pd.DataFrame, type_filter=None):
    source = df if type_filter is None else df[df["Question_Type"].isin(type_filter)]
    records = []
    for level in LEVEL_ORDER:
        for subject in SUBJECT_ORDER:
            g = source[(source["Level"] == level) & (source["Subject"] == subject)]
            if len(g) == 0:
                continue

            local = []
            for m1, m2 in itertools.combinations(MODELS, 2):
                a = g[m1].to_numpy(dtype=int)
                b = g[m2].to_numpy(dtype=int)

                both_correct = int(((a == 1) & (b == 1)).sum())
                m1_only_correct = int(((a == 1) & (b == 0)).sum())
                m2_only_correct = int(((a == 0) & (b == 1)).sum())
                both_incorrect = int(((a == 0) & (b == 0)).sum())

                tab = np.array(
                    [[both_correct, m1_only_correct],
                     [m2_only_correct, both_incorrect]],
                    dtype=int,
                )
                test = mcnemar(tab, exact=True)
                diff_pp = (a.mean() - b.mean()) * 100

                local.append({
                    "Level": level,
                    "Subject": subject,
                    "n_questions": int(len(g)),
                    "Model_1": m1,
                    "Model_2": m2,
                    "Model1_only_correct": m1_only_correct,
                    "Model2_only_correct": m2_only_correct,
                    "Both_correct": both_correct,
                    "Both_incorrect": both_incorrect,
                    "Accuracy_difference_pp_M1_minus_M2": diff_pp,
                    "McNemar_exact_P": float(test.pvalue),
                })

            if local:
                adj = multipletests(
                    [x["McNemar_exact_P"] for x in local],
                    method="holm",
                )[1]
                for rec, p_adj in zip(local, adj):
                    rec["Holm_adjusted_P_within_stratum"] = float(p_adj)
                    rec["Significance"] = p_mark(p_adj)
                records.extend(local)

    return pd.DataFrame(records)


def compute_glmm(df: pd.DataFrame, type_filter=None, maxiter=500):
    """
    Bayesian binomial mixed-effects logistic regression per Level x Subject:
        Correct ~ Model + (1 | ID)

    statsmodels BinomialBayesMixedGLM uses mean-field variational Bayes here.
    We report approximate posterior ORs and 95% credible intervals.
    """
    source = df if type_filter is None else df[df["Question_Type"].isin(type_filter)]
    records = []
    diagnostic = []

    for level in LEVEL_ORDER:
        for subject in SUBJECT_ORDER:
            g = source[(source["Level"] == level) & (source["Subject"] == subject)]
            if len(g) == 0:
                continue

            long = g[["ID"] + MODELS].melt(
                id_vars=["ID"],
                value_vars=MODELS,
                var_name="Model",
                value_name="Correct",
            )
            long["Model"] = pd.Categorical(
                long["Model"], categories=MODELS, ordered=True
            )
            long["Correct"] = long["Correct"].astype(int)

            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    model = BinomialBayesMixedGLM.from_formula(
                        "Correct ~ C(Model)",
                        {"Question_ID_random_intercept": "0 + C(ID)"},
                        long,
                    )
                    fit = model.fit_vb(
                        minim_opts={"maxiter": maxiter},
                        verbose=False,
                    )

                success = bool(getattr(fit, "optim_retvals", {}).get("success", False))
                diagnostic.append({
                    "Level": level,
                    "Subject": subject,
                    "n_questions": int(len(g)),
                    "Converged": success,
                    "Warnings": " | ".join(str(w.message) for w in caught),
                    "Random_intercept_SD": float(np.exp(fit.vcp_mean[0])),
                })

                # Fixed effects: intercept is the first term and the remaining
                # five terms compare each model with the first model (Kimi K2).
                fe_mean = np.asarray(fit.fe_mean, dtype=float)
                fe_sd = np.asarray(fit.fe_sd, dtype=float)
                beta = {MODELS[0]: 0.0}
                beta_sd = {MODELS[0]: 0.0}
                for idx, m in enumerate(MODELS[1:], start=1):
                    beta[m] = float(fe_mean[idx])
                    beta_sd[m] = float(fe_sd[idx])

                for m1, m2 in itertools.combinations(MODELS, 2):
                    log_or = beta[m1] - beta[m2]
                    # Mean-field VB approximates posterior covariance as zero.
                    se_approx = math.sqrt(beta_sd[m1] ** 2 + beta_sd[m2] ** 2)
                    low_log = log_or - 1.96 * se_approx
                    high_log = log_or + 1.96 * se_approx
                    records.append({
                        "Level": level,
                        "Subject": subject,
                        "n_questions": int(len(g)),
                        "Model_1": m1,
                        "Model_2": m2,
                        "Odds_ratio_M1_vs_M2": float(np.exp(log_or)),
                        "Approx_95%_CrI_low": float(np.exp(low_log)),
                        "Approx_95%_CrI_high": float(np.exp(high_log)),
                        "CrI_excludes_1": bool((low_log > 0) or (high_log < 0)),
                        "Random_intercept_SD": float(np.exp(fit.vcp_mean[0])),
                        "Converged": success,
                    })

            except Exception as exc:
                diagnostic.append({
                    "Level": level,
                    "Subject": subject,
                    "n_questions": int(len(g)),
                    "Converged": False,
                    "Warnings": f"FIT FAILED: {type(exc).__name__}: {exc}",
                    "Random_intercept_SD": np.nan,
                })

    return pd.DataFrame(records), pd.DataFrame(diagnostic)


def compute_overall_accuracy(df: pd.DataFrame):
    records = []
    for m in MODELS:
        z = add_accuracy_fields(df, m)
        records.append({
            "Model": m,
            "n_questions": z["n"],
            "Correct_n": z["correct"],
            "Accuracy_pct": z["accuracy_pct"],
            "Wilson_95CI_low_pct": z["wilson_low_pct"],
            "Wilson_95CI_high_pct": z["wilson_high_pct"],
        })
    return pd.DataFrame(records)


def compute_data_qc(df: pd.DataFrame):
    records = [
        ["Total rows/questions", len(df), "Each row is one item-level record."],
        ["Unique ID", df["ID"].nunique(), "ID is used as the random-intercept grouping variable."],
        ["Duplicate ID", int(df["ID"].duplicated().sum()), "Must be zero."],
        ["Unique Question_ID", df["Question_ID"].nunique(), "Question_ID repeats across categories/types; do not use alone as random-effect ID."],
        ["Rows with repeated Question_ID", int(df["Question_ID"].duplicated().sum()), "Repeated numbering is expected in this workbook."],
        ["Missing model outcomes", int(df[MODELS].isna().sum().sum()), "Must be zero for the paired tests in this script."],
        ["A-type questions", int((df["Question_Type"] == "A").sum()), "Used in A-only sensitivity analysis."],
        ["B-type questions", int((df["Question_Type"] == "B").sum()), "No shared-stem group identifier was supplied."],
    ]
    for level, n in df["Level"].value_counts().items():
        records.append([f"Level: {level}", int(n), "Number of question records."])
    for subject, n in df["Subject"].value_counts().items():
        records.append([f"Subject: {subject}", int(n), "Number of question records."])
    return pd.DataFrame(records, columns=["QC_item", "Value", "Interpretation"])


def make_accuracy_plot(df: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    offsets = np.linspace(-0.22, 0.22, len(MODELS))
    level_pos = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}

    for ax, subject in zip(axes.flat, SUBJECT_ORDER):
        for j, model in enumerate(MODELS):
            xs, xerr_low, xerr_high, ys = [], [], [], []
            for level in LEVEL_ORDER:
                g = df[(df["Level"] == level) & (df["Subject"] == subject)]
                z = add_accuracy_fields(g, model)
                xs.append(z["accuracy_pct"])
                xerr_low.append(z["accuracy_pct"] - z["wilson_low_pct"])
                xerr_high.append(z["wilson_high_pct"] - z["accuracy_pct"])
                ys.append(level_pos[level] + offsets[j])

            ax.errorbar(
                xs,
                ys,
                xerr=np.array([xerr_low, xerr_high]),
                fmt="o",
                capsize=2.5,
                markersize=5,
                linewidth=1,
                label=model,
            )

        ax.set_title(subject, fontsize=11, weight="bold")
        ax.set_xlim(65, 100)
        ax.set_xticks([65, 70, 75, 80, 85, 90, 95, 100])
        ax.set_yticks(range(len(LEVEL_ORDER)))
        ax.set_yticklabels([LEVEL_SHORT[x] for x in LEVEL_ORDER])
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_xlabel("Accuracy (%)")
        ax.set_ylabel("Professional level")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, 0.015), frameon=False
    )
    fig.suptitle(
        "Accuracy of six large language models across professional levels and knowledge domains",
        fontsize=14, weight="bold", y=0.98
    )
    fig.text(
        0.5, 0.075,
        "Points show accuracy; horizontal bars show 95% Wilson intervals. "
        "Intervals are descriptive; paired tests are reported in Table 2.",
        ha="center", fontsize=9
    )
    fig.tight_layout(rect=[0.03, 0.12, 0.99, 0.94])
    fig.savefig(outdir / "Figure_Table2_Accuracy_DotWhisker.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "Figure_Table2_Accuracy_DotWhisker.pdf", bbox_inches="tight")
    plt.close(fig)


def make_glmm_forest_plot(glmm_df: pd.DataFrame, outdir: Path):
    if glmm_df.empty:
        return
    # For readability, show comparisons against Kimi K2 only.
    d = glmm_df[glmm_df["Model_2"] == MODELS[0]].copy()
    # In the pair generation order, Kimi is Model_1 only in pairs where it appears first.
    # Reverse those rows so the display is "other model vs Kimi".
    d = d[d["Model_1"] != MODELS[0]].copy()
    if d.empty:
        # Select Kimi-vs-other rows and invert the OR.
        d = glmm_df[glmm_df["Model_1"] == MODELS[0]].copy()
        d["Model_1"], d["Model_2"] = d["Model_2"], d["Model_1"]
        old_or = d["Odds_ratio_M1_vs_M2"].copy()
        old_low = d["Approx_95%_CrI_low"].copy()
        old_high = d["Approx_95%_CrI_high"].copy()
        d["Odds_ratio_M1_vs_M2"] = 1 / old_or
        d["Approx_95%_CrI_low"] = 1 / old_high
        d["Approx_95%_CrI_high"] = 1 / old_low

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    offsets = np.linspace(-0.20, 0.20, len(MODELS) - 1)
    level_pos = {lvl: i for i, lvl in enumerate(LEVEL_ORDER)}
    comparison_models = MODELS[1:]

    for ax, subject in zip(axes.flat, SUBJECT_ORDER):
        sub = d[d["Subject"] == subject]
        for j, m in enumerate(comparison_models):
            rows = sub[sub["Model_1"] == m].set_index("Level")
            xs, lo, hi, ys = [], [], [], []
            for level in LEVEL_ORDER:
                if level not in rows.index:
                    continue
                r = rows.loc[level]
                # If duplicate levels occur unexpectedly, use first row.
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                xs.append(float(r["Odds_ratio_M1_vs_M2"]))
                lo.append(float(r["Odds_ratio_M1_vs_M2"]) - float(r["Approx_95%_CrI_low"]))
                hi.append(float(r["Approx_95%_CrI_high"]) - float(r["Odds_ratio_M1_vs_M2"]))
                ys.append(level_pos[level] + offsets[j])
            if xs:
                ax.errorbar(
                    xs, ys, xerr=np.array([lo, hi]), fmt="o",
                    capsize=2.5, markersize=5, linewidth=1, label=m
                )

        ax.axvline(1, linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_xlim(0.25, 4)
        ax.set_xticks([0.25, 0.5, 1, 2, 4])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_yticks(range(len(LEVEL_ORDER)))
        ax.set_yticklabels([LEVEL_SHORT[x] for x in LEVEL_ORDER])
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.set_title(subject, fontsize=11, weight="bold")
        ax.set_xlabel("Odds ratio vs Kimi K2 (log scale)")
        ax.set_ylabel("Professional level")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.015), frameon=False)
    fig.suptitle(
        "Pairwise mixed-effects logistic model estimates versus Kimi K2",
        fontsize=14, weight="bold", y=0.98
    )
    fig.text(
        0.5, 0.075,
        "Odds ratio >1 indicates higher odds of a correct answer than Kimi K2. "
        "Intervals are approximate 95% posterior credible intervals from variational Bayes.",
        ha="center", fontsize=9
    )
    fig.tight_layout(rect=[0.03, 0.12, 0.99, 0.94])
    fig.savefig(outdir / "FigureS1_GLMM_OR_Forest.png", dpi=300, bbox_inches="tight")
    fig.savefig(outdir / "FigureS1_GLMM_OR_Forest.pdf", bbox_inches="tight")
    plt.close(fig)


def export_results_xlsx(output_file: Path, sheets: dict[str, pd.DataFrame], note_rows: list[list]):
    # Standard pandas Excel export for portability.
    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=sheet_name[:31], index=False)
            ws = writer.sheets[sheet_name[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            for cell in ws[1]:
                from copy import copy
                from openpyxl.styles import PatternFill
                f = copy(cell.font)
                f.bold = True
                f.color = "FFFFFF"
                cell.font = f
                cell.fill = PatternFill(fill_type="solid", fgColor="1F4E78")
                a = copy(cell.alignment)
                a.wrap_text = True
                a.vertical = "center"
                cell.alignment = a
            ws.row_dimensions[1].height = 32
            for col_cells in ws.columns:
                max_len = max(
                    (len(str(c.value)) if c.value is not None else 0)
                    for c in col_cells[: min(len(col_cells), 100)]
                )
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(max_len + 2, 12), 36)

        ws = writer.book.create_sheet("Readme")
        ws.append(["Analysis notes"])
        for row in note_rows:
            ws.append(row)
        ws.freeze_panes = "A2"
        ws.column_dimensions["A"].width = 34
        ws.column_dimensions["B"].width = 110
        for cell in ws[1]:
            from copy import copy
            from openpyxl.styles import PatternFill
            f = copy(cell.font)
            f.bold = True
            f.color = "FFFFFF"
            cell.font = f
            cell.fill = PatternFill(fill_type="solid", fgColor="1F4E78")


def main():
    parser = argparse.ArgumentParser(description="Paired six-LLM accuracy analysis")
    parser.add_argument("--input", default="data_table2.xlsx", help="Input Excel file")
    parser.add_argument("--output_dir", default="table2_outputs", help="Output directory")
    args = parser.parse_args()

    input_file = Path(args.input).expanduser().resolve()
    outdir = Path(args.output_dir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_data(input_file)

    # Main analysis: all A- and B-type items.
    table2 = compute_cochran_table(df)
    mcnemar_df = compute_mcnemar(df)
    glmm_df, glmm_diag = compute_glmm(df, maxiter=500)
    overall_df = compute_overall_accuracy(df)
    qc_df = compute_data_qc(df)

    # Sensitivity analysis: A-type items only.
    a_only = df[df["Question_Type"] == "A"].copy()
    a_sensitivity = compute_cochran_table(df, type_filter=["A"])
    a_mcnemar = compute_mcnemar(df, type_filter=["A"])

    # Figures.
    make_accuracy_plot(df, outdir)
    make_glmm_forest_plot(glmm_df, outdir)

    # Notes for users/manuscript interpretation.
    notes = [
        ["Input data", str(input_file)],
        ["Main paired analysis", "Cochran's Q within each Level x Subject stratum; 12 omnibus P values adjusted by Holm."],
        ["Pairwise analysis", "Exact McNemar tests for all 15 model pairs within each stratum; Holm correction across 15 tests within each stratum."],
        ["Mixed model", "BinomialBayesMixedGLM: Correct ~ Model + (1|ID), fitted separately within each Level x Subject stratum."],
        ["GLMM interval", "Approximate 95% posterior credible interval from mean-field variational Bayes; not a conventional maximum-likelihood GLMM 95% confidence interval."],
        ["Unique item ID", "ID is unique across 3913 rows and is used for the question random intercept. Question_ID repeats across domains/types."],
        ["B-type questions", "The file has no shared-stem group ID. Main analysis includes B-type items as individual item records; A-only sensitivity analysis is provided."],
        ["Accuracy figure", "Points show accuracy; horizontal bars show 95% Wilson intervals. These are descriptive intervals; paired tests provide inferential comparisons."],
        ["Model naming", "The supplied workbook labels the Gemini column 'gemini flash 2.5'. Verify the exact deployed model/version against API records before manuscript submission."],
        ["All-six incorrect", "Descriptive count of questions answered incorrectly by all six models; do not automatically label these as cognitive blind spots."],
    ]

    sheets = {
        "Table2_Main": table2,
        "GLMM_Pairwise": glmm_df,
        "GLMM_Diagnostics": glmm_diag,
        "McNemar_Pairwise": mcnemar_df,
        "A_only_Sensitivity": a_sensitivity,
        "A_only_McNemar": a_mcnemar,
        "Overall_Accuracy": overall_df,
        "Data_QC": qc_df,
    }
    output_xlsx = outdir / "Table2_Statistical_Results.xlsx"
    export_results_xlsx(output_xlsx, sheets, notes)

    # CSVs make the individual tables easy to import into manuscripts/statistical software.
    for name, frame in sheets.items():
        frame.to_csv(outdir / f"{name}.csv", index=False, encoding="utf-8-sig")

    print("\nAnalysis completed.")
    print(f"Input: {input_file}")
    print(f"Output folder: {outdir}")
    print(f"Main table: {output_xlsx}")
    print(f"Figures: Figure_Table2_Accuracy_DotWhisker.png/.pdf; FigureS1_GLMM_OR_Forest.png/.pdf")
    print("\nCochran's Q results (Holm-adjusted across 12 strata):")
    print(table2[["Level", "Subject", "n_questions", "Cochran_Q", "P_value", "Holm_adjusted_P", "Significance"]].to_string(index=False))
    if not glmm_diag.empty and not glmm_diag["Converged"].all():
        print("\nWARNING: At least one GLMM did not converge. Inspect GLMM_Diagnostics.")
    if not glmm_df.empty:
        print(f"\nGLMM pairwise contrast rows: {len(glmm_df)}")
    print("\nImportant: review the GLMM variational-Bayes caveat and exact Gemini model name before manuscript reporting.")


if __name__ == "__main__":
    main()
