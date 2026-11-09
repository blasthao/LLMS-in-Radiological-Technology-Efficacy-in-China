#!/usr/bin/env python
# Reclassification and descriptive reanalysis of 147 baseline-selected high-error questions.
# Public input/output schema excludes original question stems and keyed answers.
#
# Main results:
#   - 2025/2026 aggregate accuracy by revised knowledge category
#   - 10,000 question-level paired bootstrap resamples for 95% intervals
#   - Distribution of questions by number of models answering correctly (0-6)
#   - Original vs revised category counts
#
# Important:
#   The questions were selected because <=1/6 baseline models answered correctly.
#   The year panels may contain different model versions. Therefore the comparison is
#   exploratory/descriptive and must not be described as evidence of model evolution.
#
# Usage:
#   python analyze_table4_revised.py --input "data_table4(3).xlsx" \
#       --output_dir Table4_Revised_Output --bootstrap 10000
#
# Dependencies:
#   artifact_tool, numpy, matplotlib

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from artifact_tool import Blob, SpreadsheetFile, Workbook


YEAR25 = [
    "2025KIMI", "2025Deepseek", "2025Qwen",
    "2025Doubao", "2025Gemini", "2025ChatGPT"
]
YEAR26 = [
    "2026kimi", "2026deepseek", "2026Qwen",
    "2026Doubao", "2026gemini", "2026chatgpt"
]

# Six revised categories. Each item receives one primary category according to
# the dominant knowledge/skill being tested by the question stem and answer.
CATEGORY_INFO = {
    "A": {
        "name": "Anatomical knowledge, anatomical localization, and radiographic positioning",
        "short": "Anatomy and positioning",
        "definition": "Normal anatomical structures and relationships, anatomical directional terms, surface landmarks, imaging anatomy, projection geometry, and patient/radiographic positioning."
    },
    "B": {
        "name": "Radiological workflows, examination protocols, image quality, and informatics systems",
        "short": "Workflow, QC and informatics",
        "definition": "Clinical examination selection and workflow, acquisition protocols, contrast-agent safety and management, quality assurance and image quality, and radiology information systems/PACS integration."
    },
    "C": {
        "name": "Quantitative technical parameters, radiation dosimetry, and radiation protection",
        "short": "Quantitative parameters and radioprotection",
        "definition": "Numerical imaging and acquisition parameters, dose quantities and calculations, dosimetry, radiation risks, tissue radiosensitivity, and radiation shielding/protection."
    },
    "D": {
        "name": "Imaging physics, equipment engineering, image reconstruction, and post-processing",
        "short": "Imaging physics and post-processing",
        "definition": "Physical interactions and image formation, imaging-system components and mechanisms, subtraction/reconstruction methods, imaging artifacts, and image-processing techniques."
    },
    "E": {
        "name": "Disease-related imaging features and diagnostic reasoning",
        "short": "Disease imaging and diagnosis",
        "definition": "Disease manifestations, disease-related imaging findings and signal characteristics, clinical diagnostic reasoning, and imaging-based disease assessment."
    },
}
CATEGORY_ORDER = ["A", "B", "C", "D", "E"]

# Targeted, auditable reassignment of items after review of stems, options, and keyed answers.
# All other records retain the user-supplied category.
RECLASSIFICATION = {
    2:   ("B", "Optical density/transparency and viewing terminology are grouped with image-quality concepts."),
    3:   ("C", "Dose-quantity relationship; principally radiation dosimetry."),
    10:  ("B", "Contrast-agent adverse reaction recognition; contrast safety."),
    17:  ("B", "The item concerns the diagnostic information process and image-viewing environment."),
    47:  ("A", "The tested decision is the initial radiographic projection/position."),
    52:  ("A", "Cranial-nerve anatomy rather than numerical/dosimetric knowledge."),
    56:  ("B", "PACS/HIS/RIS integration is part of radiology workflow and imaging informatics."),
    68:  ("D", "Hybrid subtraction is an imaging subtraction technique, rather than a general terminology item."),
    74:  ("A", "Normal pelvic-organ MR signal characteristics are treated as imaging anatomy."),
    104: ("C", "Definitions and relationships of radiation dose quantities."),
    107: ("C", "Selection of a dose metric for cancer-risk assessment."),
    108: ("A", "Anatomical orientation and directional terminology."),
    111: ("C", "Tissue radiosensitivity is a radiation-biology/protection concept."),
    113: ("C", "Radiation shielding material and protection."),
    116: ("D", "Edge sharpening is an image post-processing technique."),
    125: ("D", "Truncation artifact mitigation is an imaging artifact/technical processing issue."),
    126: ("B", "PACS access-layer devices and workstations are part of radiology informatics workflow."),
    129: ("D", "MIP is a multiplanar/3D image-display and post-processing method."),
    132: ("B", "Classification of examination procedures by invasiveness/physical harm."),
    133: ("A", "Breast developmental stages are anatomical/developmental knowledge."),
}

ORIGINAL_CATEGORY_NAMES = {
    "三维空间重构与摄影体位": "Original: 3D spatial reconstruction and radiographic positioning",
    "临床操作规范与影像质控": "Original: Clinical protocols and image quality control",
    "精确数值与剂量计算": "Original: Precise numerical values and dosimetry calculations",
    "医学概念与术语辨析": "Original: Medical concepts and terminology differentiation",
    "设备物理机制与工程原理": "Original: Physical mechanisms and engineering principles",
    "疾病病理与影像诊断学": "Original: Pathology and diagnostic imaging",
}

ORIGINAL_TO_REVISED_DEFAULT = {
    "三维空间重构与摄影体位": "A",
    "临床操作规范与影像质控": "B",
    "精确数值与剂量计算": "C",
    "医学概念与术语辨析": "B",
    "设备物理机制与工程原理": "D",
    "疾病病理与影像诊断学": "E",
}

RECLASSIFICATION_REASONS = {
    2: "Move from general terminology to workflow/image-quality concepts because the item concerns optical density, transparency, and viewing terminology.",
    3: "Move from general terminology to radiation dosimetry because the item tests dose-quantity relationships.",
    10: "Move from disease imaging to contrast-agent safety because it tests recognition of a contrast reaction.",
    17: "Move from general terminology to workflow/image-quality concepts because it concerns the diagnostic information process and viewing environment.",
    47: "Move from disease diagnosis to anatomy/positioning because it asks which radiographic projection should be performed first.",
    52: "Move from quantitative/dosimetric knowledge to anatomy because it tests cranial-nerve composition.",
    56: "Move from general terminology to radiology workflow/informatics because it tests PACS/HIS/RIS integration.",
    68: "Move from general terminology to imaging physics/techniques because hybrid subtraction is an imaging subtraction method.",
    74: "Move from disease imaging to imaging anatomy because it tests normal pelvic-organ MR signal characteristics.",
    104: "Move from professional terminology to dosimetry because the options define dose/exposure quantities.",
    107: "Move from professional terminology to dosimetry because it tests the dose quantity used for cancer-risk assessment.",
    108: "Move from professional terminology to anatomical knowledge because it tests anatomical directional planes.",
    111: "Move from professional terminology to radiation protection because it tests tissue radiosensitivity.",
    113: "Move from equipment physics to radiation protection because it tests shielding effectiveness.",
    116: "Move from clinical protocols to image processing because it tests edge-enhancement technique.",
    125: "Move from clinical protocols to imaging physics/technical processing because it tests mitigation of an imaging artifact.",
    126: "Move from general terminology to radiology workflow/informatics because it tests PACS subsystem structure.",
    129: "Move from general terminology to image reconstruction/post-processing because it tests MIP.",
    132: "Move from general terminology to clinical examination/procedure knowledge because it classifies procedure invasiveness.",
    133: "Move from general terminology to anatomical/developmental knowledge because it tests breast developmental stages.",
}

def read_input(path):
    wb = SpreadsheetFile.import_xlsx(Blob.load(str(path)))
    sheet_info = wb.inspect({"kind": "sheet", "include": "id,name"}).ndjson
    first_sheet = json.loads(sheet_info.splitlines()[0])["name"]
    ws = wb.worksheets.get_item(first_sheet)
    raw = ws.get_range("A1:W500").values
    raw = [r for r in raw if r and any(v is not None for v in r)]
    headers, rows = raw[0], raw[1:]
    required = ["ID", "Subject", "Classification"] + YEAR25 + YEAR26
    missing = [c for c in required if c not in headers]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    ix = {h: headers.index(h) for h in headers}
    records = []
    for excel_row, row in enumerate(rows, start=2):
        r = {h: row[ix[h]] for h in headers}
        if r["ID"] is None:
            raise ValueError(f"Missing ID at Excel row {excel_row}")
        for c in YEAR25 + YEAR26:
            if r[c] not in (0, 1):
                raise ValueError(f"{c} must be 0/1; row {excel_row} contains {r[c]}")
        r["ID"] = int(r["ID"])
        r["Original_Classification"] = r["Classification"]
        baseline_n = sum(int(r[c]) for c in YEAR25)
        r["Correct_2025_n_models"] = baseline_n
        r["Correct_2026_n_models"] = sum(int(r[c]) for c in YEAR26)
        if baseline_n > 1:
            raise ValueError(
                f"ID {r['ID']} has {baseline_n} correct 2025 model(s); "
                "the current workbook should contain only 0/6 or 1/6 items."
            )
        revised_code = ORIGINAL_TO_REVISED_DEFAULT.get(r["Original_Classification"])
        if revised_code is None:
            raise ValueError(f"Unknown original category: {r['Original_Classification']}")
        reason = ""
        if r["ID"] in RECLASSIFICATION:
            revised_code, reason = RECLASSIFICATION[r["ID"]]
        r["Revised_Category_Code"] = revised_code
        r["Revised_Category"] = CATEGORY_INFO[revised_code]["name"]
        r["Reclassification_Reason"] = reason
        r["Reclassified"] = bool(reason)
        records.append(r)

    ids = [r["ID"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate ID detected.")
    return records, headers, wb, first_sheet


def item_bootstrap(y25, y26, n_boot=10000, seed=20261105):
    """
    Resample whole question rows within category.
    All six 2025 and six 2026 model outcomes stay together for a sampled item.
    """
    y25 = np.asarray(y25, dtype=float)
    y26 = np.asarray(y26, dtype=float)
    item25 = y25.mean(axis=1) * 100.0
    item26 = y26.mean(axis=1) * 100.0
    itemdiff = item26 - item25
    n = len(item25)

    p25 = float(item25.mean())
    p26 = float(item26.mean())
    d = float(itemdiff.mean())

    rng = np.random.default_rng(seed)
    b25 = np.empty(n_boot, dtype=float)
    b26 = np.empty(n_boot, dtype=float)
    bd = np.empty(n_boot, dtype=float)
    batch = 500
    pos = 0
    while pos < n_boot:
        m = min(batch, n_boot - pos)
        ix = rng.integers(0, n, size=(m, n))
        b25[pos:pos+m] = item25[ix].mean(axis=1)
        b26[pos:pos+m] = item26[ix].mean(axis=1)
        bd[pos:pos+m] = itemdiff[ix].mean(axis=1)
        pos += m

    ci25 = np.percentile(b25, [2.5, 97.5])
    ci26 = np.percentile(b26, [2.5, 97.5])
    cid = np.percentile(bd, [2.5, 97.5])

    return {
        "n_questions": n,
        "N_responses": n * 6,
        "Correct_2025": int(y25.sum()),
        "Correct_2026": int(y26.sum()),
        "Accuracy_2025_pct": p25,
        "Accuracy_2025_CI_low": float(ci25[0]),
        "Accuracy_2025_CI_high": float(ci25[1]),
        "Accuracy_2026_pct": p26,
        "Accuracy_2026_CI_low": float(ci26[0]),
        "Accuracy_2026_CI_high": float(ci26[1]),
        "Difference_2026_minus_2025_pp": d,
        "Difference_CI_low": float(cid[0]),
        "Difference_CI_high": float(cid[1]),
        "2025_nN_pct": f'{int(y25.sum())}/{n*6} ({p25:.1f}%)',
        "2026_nN_pct": f'{int(y26.sum())}/{n*6} ({p26:.1f}%)',
        "Difference_with_CI": f'{d:+.1f} pp ({cid[0]:+.1f}, {cid[1]:+.1f})',
    }


def summarize_categories(records, category_field, n_boot=10000, seed_offset=0):
    rows = []
    counts = []
    for code in CATEGORY_ORDER:
        catname = CATEGORY_INFO[code]["name"]
        group = [r for r in records if r[category_field] == catname]
        if not group:
            continue
        y25 = np.asarray([[int(r[c]) for c in YEAR25] for r in group], dtype=int)
        y26 = np.asarray([[int(r[c]) for c in YEAR26] for r in group], dtype=int)
        summary = item_bootstrap(
            y25, y26, n_boot=n_boot,
            seed=20261105 + seed_offset + CATEGORY_ORDER.index(code)*53
        )
        total = len(records)
        summary.update({
            "Category_Code": code,
            "Category": catname,
            "Short_Category": CATEGORY_INFO[code]["short"],
            "Definition": CATEGORY_INFO[code]["definition"],
            "Questions_n_pct": f'{len(group)} ({100*len(group)/total:.1f}%)',
            "Questions_pct": 100*len(group)/total,
        })
        rows.append(summary)

        # Distribution across items according to number of models correct (0-6).
        for year, arr in [(2025, y25), (2026, y26)]:
            n_correct = arr.sum(axis=1)
            for k in range(7):
                n_items = int(np.sum(n_correct == k))
                counts.append({
                    "Category_Code": code,
                    "Category": catname,
                    "Year": year,
                    "Correct_models_count": k,
                    "Question_count": n_items,
                    "Question_percent": 100*n_items/len(group),
                    "n_questions": len(group),
                })
    return rows, counts


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def make_dumbbell(summary_rows, outdir):
    ordered = list(reversed(summary_rows))
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    y = np.arange(len(ordered))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    c25, c26 = colors[0], colors[1]

    for i, r in enumerate(ordered):
        y0 = y[i]
        a25 = r["Accuracy_2025_pct"]
        a26 = r["Accuracy_2026_pct"]
        ax.plot([a25, a26], [y0, y0], color="0.65", linewidth=1.4, zorder=1)
        ax.errorbar(
            a25, y0-0.11,
            xerr=[[a25-r["Accuracy_2025_CI_low"]], [r["Accuracy_2025_CI_high"]-a25]],
            fmt="o", color=c25, markersize=7, capsize=3, linewidth=1.5, zorder=3,
            label="2025 baseline panel" if i == 0 else None
        )
        ax.errorbar(
            a26, y0+0.11,
            xerr=[[a26-r["Accuracy_2026_CI_low"]], [r["Accuracy_2026_CI_high"]-a26]],
            fmt="s", color=c26, markersize=7, capsize=3, linewidth=1.5, zorder=3,
            label="2026 model panel" if i == 0 else None
        )
        ax.text(
            1.015, y0,
            f'Δ {r["Difference_2026_minus_2025_pp"]:+.1f} pp\n'
            f'95% item-bootstrap CI {r["Difference_CI_low"]:+.1f} to {r["Difference_CI_high"]:+.1f}',
            transform=ax.get_yaxis_transform(), ha="left", va="center",
            fontsize=8.5, clip_on=False
        )

    ax.set_yticks(y)
    ax.set_yticklabels([
        f'{r["Short_Category"]} (n={r["n_questions"]})' for r in ordered
    ], fontsize=9)
    ax.set_xlabel("Aggregate accuracy across the six-model panel (%)")
    ax.set_xlim(0, 46)
    ax.set_ylim(-0.65, len(ordered)-0.35)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title(
        "Exploratory descriptive comparison in baseline-selected high-error questions\n"
        "147 questions: 74 with 0/6 and 73 with 1/6 models correct in 2025",
        fontsize=12, weight="bold"
    )
    fig.subplots_adjust(left=0.37, right=0.73, top=0.88, bottom=0.12)
    fig.savefig(outdir / "Figure3_HighError_RevisedCategories.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / "Figure3_HighError_RevisedCategories.pdf", bbox_inches="tight")
    plt.close(fig)


def make_item_count_distribution(count_rows, outdir):
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 12.3))
    axes = axes.ravel()
    cmap = plt.get_cmap("viridis", 7)
    for j, code in enumerate(CATEGORY_ORDER):
        ax = axes[j]
        subset = [r for r in count_rows if r["Category_Code"] == code]
        catinfo = CATEGORY_INFO[code]
        group_n = next(r["n_questions"] for r in subset)
        for yi, year in enumerate([2025, 2026]):
            row = [r for r in subset if r["Year"] == year]
            left = 0.0
            for k in range(7):
                item = next(r for r in row if r["Correct_models_count"] == k)
                width = item["Question_percent"]
                ax.barh(
                    yi, width, left=left, height=0.55,
                    color=cmap(k), edgecolor="white", linewidth=0.4,
                    label=f"{k} model(s) correct" if j == 0 and yi == 0 else None
                )
                left += width
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["2025 baseline", "2026 model panel"], fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Questions (%)")
        ax.set_title(f"({chr(65+j)}) {catinfo['short']} (n={group_n})", fontsize=9.5, weight="bold")
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.012))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(outdir / "FigureS3_ItemCorrectCount_RevisedCategories.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / "FigureS3_ItemCorrectCount_RevisedCategories.pdf", bbox_inches="tight")
    plt.close(fig)


def make_category_comparison(records, outdir):
    original_order = [
        "三维空间重构与摄影体位",
        "临床操作规范与影像质控",
        "精确数值与剂量计算",
        "设备物理机制与工程原理",
        "医学概念与术语辨析",
        "疾病病理与影像诊断学",
    ]
    original_labels = [
        "3D spatial reconstruction/positioning",
        "Clinical protocols/image quality control",
        "Numerical values/dosimetry",
        "Physical mechanisms/engineering",
        "Concepts/terminology",
        "Pathology/diagnostic imaging",
    ]
    original_counts = [sum(r["Original_Classification"] == c for r in records) for c in original_order]
    revised_counts = [sum(r["Revised_Category_Code"] == c for r in records) for c in CATEGORY_ORDER]
    revised_labels = [CATEGORY_INFO[c]["short"] for c in CATEGORY_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    axes[0].barh(np.arange(6), original_counts)
    axes[0].set_yticks(np.arange(6))
    axes[0].set_yticklabels(original_labels, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Number of questions")
    axes[0].set_title("Original six-category classification", fontsize=10, weight="bold")
    axes[0].grid(axis="x", alpha=0.2)
    for i, v in enumerate(original_counts):
        axes[0].text(v+0.5, i, str(v), va="center", fontsize=8)

    axes[1].barh(np.arange(5), revised_counts)
    axes[1].set_yticks(np.arange(5))
    axes[1].set_yticklabels(revised_labels, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Number of questions")
    axes[1].set_title("Refined five-category classification", fontsize=10, weight="bold")
    axes[1].grid(axis="x", alpha=0.2)
    for i, v in enumerate(revised_counts):
        axes[1].text(v+0.5, i, str(v), va="center", fontsize=8)

    fig.suptitle("Distribution of the 147 questions before and after category refinement",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(outdir / "FigureS4_Category_Counts_Original_vs_Revised.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / "FigureS4_Category_Counts_Original_vs_Revised.pdf", bbox_inches="tight")
    plt.close(fig)


def add_sheet(wb, name, fields, rows, widths=None):
    sh = wb.worksheets.add(name)
    matrix = [fields] + [[r.get(f) for f in fields] for r in rows]
    sh.get_range_by_indexes(0, 0, len(matrix), len(fields)).values = matrix
    sh.get_range_by_indexes(0, 0, 1, len(fields)).format = {
        "fill": "#1F4E78",
        "font": {"bold": True, "color": "#FFFFFF"},
        "horizontal_alignment": "center",
        "vertical_alignment": "center",
        "wrap_text": True,
        "row_height": 34,
    }
    sh.freeze_panes.freeze_rows(1)
    sh.get_range_by_indexes(1, 0, max(1, len(rows)), len(fields)).format.wrap_text = True
    if widths:
        for i, w in enumerate(widths):
            sh.get_range_by_indexes(0, i, len(matrix), 1).format.column_width = w
    else:
        sh.get_range_by_indexes(0, 0, len(matrix), len(fields)).format.autofit_columns()
    return sh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data_table4(3).xlsx")
    parser.add_argument("--output_dir", default="Table4_Revised_Output")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    records, original_headers, source_wb, sheet_name = read_input(args.input)

    # Apply revised grouping via revised category names.
    for r in records:
        r["Original_Category_Name"] = r["Original_Classification"]
        r["Original_Category_Code"] = ORIGINAL_TO_REVISED_DEFAULT[r["Original_Classification"]]
        # Store original category label separately for comparison.
        r["Original_Category_English"] = ORIGINAL_CATEGORY_NAMES[r["Original_Classification"]]

    # Original six-category summary, retained for side-by-side documentation.
    original_summary = []
    revised_summary = []
    count_rows = []

    original_meta = [
        ("三维空间重构与摄影体位", "Original: 3D spatial reconstruction and radiographic positioning", "3D spatial reconstruction/positioning"),
        ("临床操作规范与影像质控", "Original: Clinical protocols and image quality control", "Clinical protocols/image QC"),
        ("精确数值与剂量计算", "Original: Precise numerical values and dosimetry calculations", "Numerical values/dosimetry"),
        ("设备物理机制与工程原理", "Original: Physical mechanisms and engineering principles", "Physical mechanisms/engineering"),
        ("医学概念与术语辨析", "Original: Medical concepts and terminology differentiation", "Concepts/terminology"),
        ("疾病病理与影像诊断学", "Original: Pathology and diagnostic imaging", "Pathology/diagnostic imaging"),
    ]
    for oi, (orig_cat, orig_name, orig_short) in enumerate(original_meta):
        group = [r for r in records if r["Original_Classification"] == orig_cat]
        if group:
            y25 = np.asarray([[int(r[c]) for c in YEAR25] for r in group], dtype=int)
            y26 = np.asarray([[int(r[c]) for c in YEAR26] for r in group], dtype=int)
            s = item_bootstrap(y25, y26, n_boot=args.bootstrap,
                               seed=20261105+40000+oi*53)
            s.update({
                "Category_Code": f"Original_{oi+1}",
                "Category": orig_name,
                "Short_Category": orig_short,
                "n_questions": len(group),
                "Questions_n_pct": f'{len(group)} ({100*len(group)/len(records):.1f}%)',
                "Mapping": "Original"
            })
            original_summary.append(s)

    # revised grouping
    for code in CATEGORY_ORDER:
        group = [r for r in records if r["Revised_Category_Code"] == code]
        if not group:
            continue
        y25 = np.asarray([[int(r[c]) for c in YEAR25] for r in group], dtype=int)
        y26 = np.asarray([[int(r[c]) for c in YEAR26] for r in group], dtype=int)
        s = item_bootstrap(y25, y26, n_boot=args.bootstrap,
                           seed=20261105+10000+CATEGORY_ORDER.index(code)*53)
        s.update({
            "Category_Code": code,
            "Category": CATEGORY_INFO[code]["name"],
            "Short_Category": CATEGORY_INFO[code]["short"],
            "Definition": CATEGORY_INFO[code]["definition"],
            "n_questions": len(group),
            "Questions_n_pct": f'{len(group)} ({100*len(group)/len(records):.1f}%)',
            "Questions_pct": 100*len(group)/len(records),
            "Mapping": "Revised"
        })
        revised_summary.append(s)

        for year, cols in [(2025, YEAR25), (2026, YEAR26)]:
            arr = np.asarray([[int(r[c]) for c in cols] for r in group], dtype=int)
            nums = arr.sum(axis=1)
            for k in range(7):
                count_rows.append({
                    "Category_Code": code,
                    "Category": CATEGORY_INFO[code]["name"],
                    "Year": year,
                    "Correct_models_count": k,
                    "Question_count": int(np.sum(nums == k)),
                    "Question_percent": float(np.mean(nums == k)*100),
                    "n_questions": len(group)
                })

    # Reclassification audit table.
    change_rows = []
    for r in records:
        if r["Reclassified"]:
            change_rows.append({
                "ID": r["ID"],
                "Subject": r["Subject"],
                "Original_category": r["Original_Classification"],
                "Revised_category": r["Revised_Category"],
                "Reason": r["Reclassification_Reason"],
                "Correct_2025_n_models": r["Correct_2025_n_models"],
                "Correct_2026_n_models": r["Correct_2026_n_models"],
            })

    # Per-item revised classification and outcomes.
    item_rows = []
    for r in records:
        item_rows.append({
            "ID": r["ID"],
            "Subject": r["Subject"],
            "Original_category": r["Original_Classification"],
            "Revised_category_code": r["Revised_Category_Code"],
            "Revised_category": r["Revised_Category"],
            "Reclassification_reason": r["Reclassification_Reason"],
            "Correct_2025_n_models": r["Correct_2025_n_models"],
            "Correct_2026_n_models": r["Correct_2026_n_models"],
            "Reclassified": r["Reclassified"],
        })

    make_dumbbell(revised_summary, outdir)
    make_item_count_distribution(count_rows, outdir)
    make_category_comparison(records, outdir)

    # Output CSVs.
    main_fields = [
        "Category_Code", "Category", "Definition", "Questions_n_pct", "n_questions", "N_responses",
        "Correct_2025", "2025_nN_pct",
        "Accuracy_2025_pct", "Accuracy_2025_CI_low", "Accuracy_2025_CI_high",
        "Correct_2026", "2026_nN_pct",
        "Accuracy_2026_pct", "Accuracy_2026_CI_low", "Accuracy_2026_CI_high",
        "Difference_2026_minus_2025_pp", "Difference_CI_low", "Difference_CI_high", "Difference_with_CI"
    ]
    original_fields = [
        "Category_Code", "Category", "Short_Category", "Mapping", "n_questions", "N_responses",
        "Correct_2025", "2025_nN_pct", "Accuracy_2025_pct",
        "Correct_2026", "2026_nN_pct", "Accuracy_2026_pct",
        "Difference_2026_minus_2025_pp", "Difference_with_CI"
    ]
    count_fields = ["Category_Code", "Category", "Year", "Correct_models_count", "Question_count", "Question_percent", "n_questions"]
    change_fields = ["ID", "Subject", "Original_category", "Revised_category", "Reason", "Correct_2025_n_models", "Correct_2026_n_models"]
    item_fields = ["ID", "Subject", "Original_category", "Revised_category_code", "Revised_category", "Reclassification_reason", "Correct_2025_n_models", "Correct_2026_n_models", "Reclassified"]

    write_csv(outdir/"Table4_Revised_Main.csv", revised_summary, main_fields)
    write_csv(outdir/"Table4_Original_Category_Summary.csv", original_summary, original_fields)
    write_csv(outdir/"FigureS3_ItemCorrectCount_Data.csv", count_rows, count_fields)
    write_csv(outdir/"Classification_Changes.csv", change_rows, change_fields)
    write_csv(outdir/"Item_Level_Revised_Classification.csv", item_rows, item_fields)

    qc_rows = [
        {"QC_item": "Question count", "Value": len(records), "Note": "Updated workbook after removing ID 95"},
        {"QC_item": "Unique IDs", "Value": len({r['ID'] for r in records}), "Note": "Should equal question count"},
        {"QC_item": "2025 selection criterion", "Value": "All 147 rows pass", "Note": "Each row has 0 or 1 correct answer among six baseline models"},
        {"QC_item": "2025 baseline correct-count distribution", "Value": "74 questions at 0/6; 73 questions at 1/6", "Note": "Rechecked from model-level binary outcomes"},
        {"QC_item": "Reclassified questions", "Value": len(change_rows), "Note": "See Classification_Changes"},
        {"QC_item": "Revised category totals", "Value": "; ".join("{}={}".format(r["Category_Code"], r["n_questions"]) for r in revised_summary), "Note": "Categories are mutually exclusive and sum to 147"},
        {"QC_item": "Inferential year-to-year P values", "Value": "Not calculated", "Note": "Outcome-selected subset and non-version-matched model panels; use exploratory descriptive estimates only"},
        {"QC_item": "Bootstrap unit", "Value": "Question row", "Note": "Six 2025 and six 2026 model responses remain together within each sampled item"},
        {"QC_item": "Shared-stem IDs", "Value": "Not supplied", "Note": "If multiple items share a case/stem, cluster bootstrap by validated parent stem is preferable"},
    ]
    workbook = Workbook.create()
    add_sheet(workbook, "Table4_Revised_Main", main_fields, revised_summary,
              widths=[10, 58, 88, 18, 14, 14, 14, 22, 18, 20, 20, 14, 22, 18, 20, 20, 20, 18, 18, 30])
    add_sheet(workbook, "Original_Category_Summary", original_fields, original_summary,
              widths=[10, 58, 30, 14, 14, 14, 14, 22, 18, 14, 22, 18, 20, 30])
    add_sheet(workbook, "Classification_Changes", change_fields, change_rows,
              widths=[10, 35, 30, 70, 88, 18, 18])
    add_sheet(workbook, "Item_Level_Revised", item_fields, item_rows,
              widths=[10, 35, 30, 18, 70, 88, 18, 18, 14])
    add_sheet(workbook, "Item_Correct_Counts", count_fields, count_rows,
              widths=[10, 70, 12, 20, 18, 18, 14])
    add_sheet(workbook, "Data_QC", ["QC_item", "Value", "Note"], qc_rows,
              widths=[42, 45, 95])
    readme_rows = [
        {"Item": "Selection", "Description": "147 items: 74 with 0/6 baseline models correct and 73 with 1/6 correct; ID 95 has been removed."},
        {"Item": "Revised categories", "Description": "A anatomy/localization/positioning; B radiological workflow, examination protocols, image quality and informatics; C quantitative technical parameters/dosimetry/radiation protection; D imaging physics/equipment/subtraction/reconstruction/post-processing; E disease imaging/diagnostic reasoning. The former small terminology/informatics class is assigned by the specific competency tested."},
        {"Item": "Reclassification", "Description": f"{len(change_rows)} IDs reassigned; all changes and rationales are listed in Classification_Changes and Item_Level_Revised."},
        {"Item": "Accuracy", "Description": "Aggregate accuracy = total correct model responses / (number of questions x 6)."},
        {"Item": "Bootstrap", "Description": f"{args.bootstrap} whole-question row resamples per category; each item's 6+6 model responses remain together."},
        {"Item": "Interpretation", "Description": "Conditional descriptive comparison only. Baseline-selected items and non-version-matched model panels cannot establish within-model evolution."},
        {"Item": "P values", "Description": "No Pearson chi-square or cross-year P values are reported."},
        {"Item": "Figure 3", "Description": "Dumbbell plot of 2025 and 2026 aggregate accuracy, item-bootstrap intervals, and percentage-point differences."},
        {"Item": "Figure S3", "Description": "Question-level distribution by number of correct models (0-6) in each year and revised category."},
        {"Item": "Figure S4", "Description": "Original versus revised category counts."},
    ]
    add_sheet(workbook, "Readme", ["Item", "Description"], readme_rows, widths=[28, 130])

    xlsx_path = outdir/"Table4_Revised_Analysis.xlsx"
    SpreadsheetFile.export_xlsx(workbook).save(str(xlsx_path))

    print(f"Completed: n={len(records)}; 2025 correctness distribution=74 at 0/6, 73 at 1/6.")
    print(f"Reclassified items: {len(change_rows)}")
    print("\nRevised category results:")
    for r in revised_summary:
        print(
            f'{r["Short_Category"]}: n={r["n_questions"]}; '
            f'2025={r["2025_nN_pct"]}; 2026={r["2026_nN_pct"]}; '
            f'difference={r["Difference_with_CI"]}'
        )
    print("\nRevised category counts:", {r["Category_Code"]: r["n_questions"] for r in revised_summary})
    print("Workbook:", xlsx_path)
    print("Figures: Figure3_HighError_RevisedCategories, FigureS3_ItemCorrectCount_RevisedCategories, FigureS4_Category_Counts_Original_vs_Revised")


if __name__ == "__main__":
    main()
