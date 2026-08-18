import os
import pandas as pd
import numpy as np
from statsmodels.stats.multitest import multipletests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_FILE = os.path.join(SCRIPT_DIR, "..", "data", "final_whole_summary.csv")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")

METRIC = "exact_acc"
METHOD_FISER = "FISER"
METHOD_CEMIR = "CEMIR"
VALID_REPS = {0, 1, 2}

MODELS_MAIN = ["GPT_5", "GPT_5_MINI", "GPT_5_NANO", "Qwen_3b"]
LEVELS = [1, 2, 3, 4]

N_BOOT = 2000
N_PERM = 20000
SEED = 42


def episode_level_fractions(df, id_cols, method_col, metric_col,
                             method_a, method_b, rep_col="repetition"):
    sub = df[df[rep_col].isin(VALID_REPS)].copy()

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


def build_table1(df):
    results = []
    for model in MODELS_MAIN:
        sub = df[df["model"] == model]
        frac_df, raw_f, raw_c = episode_level_fractions(
            sub, ["episode_id"], "method", METRIC, METHOD_FISER, METHOD_CEMIR
        )
        d = (frac_df[f"{METHOD_CEMIR}_frac"] - frac_df[f"{METHOD_FISER}_frac"]).values
        test = cluster_paired_test(d)
        test["FISER_SR%"] = round(raw_f.mean() * 100, 2)
        test["CEMIR_SR%"] = round(raw_c.mean() * 100, 2)
        test["Model"] = model
        results.append(test)
    return apply_holm(pd.DataFrame(results))


def build_table2(df):
    results = []
    for level in LEVELS:
        sub = df[df["level"] == level]
        frac_df, raw_f, raw_c = episode_level_fractions(
            sub, ["model", "episode_id"], "method", METRIC, METHOD_FISER, METHOD_CEMIR
        )
        d = (frac_df[f"{METHOD_CEMIR}_frac"] - frac_df[f"{METHOD_FISER}_frac"]).values
        test = cluster_paired_test(d)
        test["FISER_SR%"] = round(raw_f.mean() * 100, 2)
        test["CEMIR_SR%"] = round(raw_c.mean() * 100, 2)
        test["Level"] = level
        results.append(test)
    return apply_holm(pd.DataFrame(results))


def build_table3(df):
    results = []
    for model in MODELS_MAIN:
        for level in LEVELS:
            sub = df[(df["model"] == model) & (df["level"] == level)]
            frac_df, raw_f, raw_c = episode_level_fractions(
                sub, ["episode_id"], "method", METRIC, METHOD_FISER, METHOD_CEMIR
            )
            if len(frac_df) == 0:
                continue
            d = (frac_df[f"{METHOD_CEMIR}_frac"] - frac_df[f"{METHOD_FISER}_frac"]).values
            test = cluster_paired_test(d)
            test["FISER_SR%"] = round(raw_f.mean() * 100, 2)
            test["CEMIR_SR%"] = round(raw_c.mean() * 100, 2)
            test["Model"] = model
            test["Level"] = level
            results.append(test)
    return apply_holm(pd.DataFrame(results))


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    main_df = pd.read_csv(MAIN_FILE)
    main_df["method"] = main_df["method"].astype(str).str.strip()
    main_df["model"] = main_df["model"].astype(str).str.strip()

    print("\n=== TABLE 1 (per model, all levels pooled) ===")
    t1 = build_table1(main_df)
    print(t1.to_string(index=False))
    t1.to_csv(os.path.join(RESULTS_DIR, "table1_cluster.csv"), index=False)

    print("\n=== TABLE 2 (per level, all models pooled) ===")
    t2 = build_table2(main_df)
    print(t2.to_string(index=False))
    t2.to_csv(os.path.join(RESULTS_DIR, "table2_cluster.csv"), index=False)

    print("\n=== TABLE 3 (per model x level) ===")
    t3 = build_table3(main_df)
    print(t3.to_string(index=False))
    t3.to_csv(os.path.join(RESULTS_DIR, "table3_cluster.csv"), index=False)

    print("\nDone. SR%/Delta match the original raw (attempt-level) numbers.")
    print("p-values/CIs come from cluster-level permutation + bootstrap,")
    print("treating each episode (not each repetition) as independent.")
