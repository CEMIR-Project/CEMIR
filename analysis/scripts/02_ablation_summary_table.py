import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV = os.path.join(SCRIPT_DIR, "..", "data", "ablation_customized_new_Qwen.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

MODALITY_ORDER = [
    "Raw Text + Image",
    "Raw Text + Enhanced Text",
]

METRIC_COL = "exact_acc"


def se_binary(values):
    n = len(values)
    if n < 2:
        return np.nan
    p = values.mean()
    return np.sqrt(p * (1 - p) / n)


def main():
    df = pd.read_csv(INPUT_CSV)
    df = df[df["model"].notna()]
    df = df[~df["model"].isin(["model"])]
    df = df[df["modality"].notna()]
    df = df[~df["modality"].isin(["method"])]
    for col in ["level", METRIC_COL, "repetition"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["level", METRIC_COL, "repetition"])
    df["level"] = df["level"].astype(int)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    levels = sorted(df["level"].unique())

    table_rows = []
    for level in levels:
        level_df = df[df["level"] == level]
        for label in MODALITY_ORDER:
            sub = level_df[level_df["modality"] == label]
            if len(sub) == 0:
                continue
            table_rows.append({
                "Modality": label,
                "Level": level,
                "Exact Accuracy (%)": round(sub[METRIC_COL].mean() * 100, 2),
                "Exact Accuracy SE": round(se_binary(sub[METRIC_COL]) * 100, 2),
            })

    table_df = pd.DataFrame(table_rows)
    table_path = os.path.join(RESULTS_DIR, "ablation_summary_table.csv")
    table_df.to_csv(table_path, index=False)
    print(f"Saved summary table: {table_path}")


if __name__ == "__main__":
    main()
