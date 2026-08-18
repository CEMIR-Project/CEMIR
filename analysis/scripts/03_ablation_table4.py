import os
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATION_FILE = os.path.join(SCRIPT_DIR, "..", "data", "ablation_customized_new_Qwen.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

METRIC = "exact_acc"
LEVELS = [1, 2, 3, 4]

ABLATION_MODEL_VALUE = "Qwen_Omni"
ABLATION_MODALITIES = ["Raw Text + Enhanced Text", "Raw Text + Image"]

N_BOOT = 2000
N_PERM = 20000
SEED = 42


def episode_level_fractions(df, id_cols, method_col, metric_col,
                             method_a, method_b, rep_col="repetition"):
    sub = df[df[rep_col].isin({0, 1, 2})].copy()

    raw_a = sub[sub[method_col] == method_a][metric_col].values
    raw_b = sub[sub[method_col] == method_b][metric_col].values

    frac = (
        sub.groupby(id_cols + [method_col])[metric_col]
        .agg(["mean", "count"])
        .reset_index()
    )
    wide_mean = frac.pivot_table(index=id_cols, columns=method_col, values="mean")

    if method_a not in wide_mean.columns:
        wide_mean[method_a] = np.nan
    if method_b not in wide_mean.columns:
        wide_mean[method_b] = np.nan

    out = wide_mean.rename(columns={method_a: f"{method_a}_frac",
                                     method_b: f"{method_b}_frac"})
    out = out.dropna(subset=[f"{method_a}_frac", f"{method_b}_frac"])

    return out, raw_a, raw_b


def cluster_paired_test(d, n_perm=N_PERM, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(d)
    observed = d.mean()

    signs = rng.choice([-1, 1], size=(n_perm, n))
    perm_means = (signs * d).mean(axis=1)
    p_value = np.mean(np.abs(perm_means) >= np.abs(observed) - 1e-12)

    boot_means = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means.append(d[idx].mean())
    ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

    return {
        "delta_pp": round(observed * 100, 2),
        "ci_low": round(ci_low * 100, 2),
        "ci_high": round(ci_high * 100, 2),
        "p_value": round(p_value, 6),
        "n_episodes": n,
    }


def apply_holm(df, pcol="p_value"):
    df = df.copy()
    _, p_holm, _, _ = multipletests(df[pcol].values, alpha=0.05, method="holm")
    df["p_holm"] = p_holm
    df["significant"] = np.where(df["p_holm"] < 0.05, "Yes", "No")
    return df


def build_table4(df):
    fiser_df = df[df["modality"] == "FISER"].copy()
    ablation_df = df[df["model"] == ABLATION_MODEL_VALUE].copy()

    results = []
    for level in LEVELS:
        fiser_level = fiser_df[fiser_df["level"] == level][
            ["episode_id", "repetition", METRIC]
        ].rename(columns={METRIC: "val"})
        fiser_level["method"] = "FISER"

        for modality in ABLATION_MODALITIES:
            mod_level = ablation_df[
                (ablation_df["level"] == level) &
                (ablation_df["modality"] == modality)
            ][["episode_id", "repetition", METRIC]].rename(columns={METRIC: "val"})
            mod_level["method"] = "MODALITY"

            combined = pd.concat([fiser_level, mod_level], ignore_index=True)
            frac_df, raw_f, raw_m = episode_level_fractions(
                combined, ["episode_id"], "method", "val", "FISER", "MODALITY"
            )
            if len(frac_df) == 0:
                continue
            d = (frac_df["MODALITY_frac"] - frac_df["FISER_frac"]).values
            test = cluster_paired_test(d)
            test["FISER_SR%"] = round(raw_f.mean() * 100, 2)
            test["Modality_SR%"] = round(raw_m.mean() * 100, 2)
            test["Level"] = level
            test["Modality"] = modality
            results.append(test)
    return apply_holm(pd.DataFrame(results))


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    ablation_df = pd.read_csv(ABLATION_FILE)
    ablation_df = ablation_df[ablation_df["model"].notna()]
    ablation_df = ablation_df[~ablation_df["model"].isin(["model"])]
    ablation_df = ablation_df[ablation_df["modality"].notna()]
    ablation_df = ablation_df[~ablation_df["modality"].isin(["method"])]
    for col in ["level", METRIC, "repetition"]:
        ablation_df[col] = pd.to_numeric(ablation_df[col], errors="coerce")
    ablation_df = ablation_df.dropna(subset=["level", METRIC, "repetition"])

    print("\n=== TABLE 4 (ablation, Qwen-2.5-3B) ===")
    t4 = build_table4(ablation_df)
    print(t4.to_string(index=False))
    t4.to_csv(os.path.join(RESULTS_DIR, "table4_cluster.csv"), index=False)

    print("\nDone. SR%/Delta match the original raw (attempt-level) numbers.")
    print("p-values/CIs come from cluster-level permutation + bootstrap,")
    print("treating each episode (not each repetition) as independent.")
