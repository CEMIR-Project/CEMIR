import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_FILE = os.path.join(SCRIPT_DIR, "..", "data", "final_whole_summary.csv")
FIGURES_DIR = os.path.join(SCRIPT_DIR, "..", "figures")

METRIC = "exact_acc"
METHOD_FISER = "FISER"
METHOD_CEMIR = "CEMIR"
VALID_REPS = {0, 1, 2}
MODELS = ["GPT_5", "GPT_5_MINI", "GPT_5_NANO", "Qwen_3b"]
MODEL_LABELS = ["GPT_5", "GPT_5_MINI", "GPT_5_NANO", "Qwen-2.5-3B"]
LEVELS = [1, 2, 3, 4]

os.makedirs(FIGURES_DIR, exist_ok=True)


def point_estimate_sr(df, model, level, method):
    sub = df[(df["model"] == model) & (df["level"] == level) &
             (df["method"] == method) & (df["repetition"].isin(VALID_REPS))]
    return sub[METRIC].mean() * 100


df = pd.read_csv(MAIN_FILE)
df["method"] = df["method"].astype(str).str.strip()
df["model"] = df["model"].astype(str).str.strip()

data = {}

for level in LEVELS:
    data[level] = {}
    for model in MODELS:
        f_pt = point_estimate_sr(df, model, level, METHOD_FISER)
        c_pt = point_estimate_sr(df, model, level, METHOD_CEMIR)
        data[level][model] = (f_pt, c_pt)
        print(f"Level {level} / {model}: FISER={f_pt:.2f}%  CEMIR={c_pt:.2f}%")


def plot_level(level, filename):
    y = np.arange(len(MODELS))
    height = 0.35

    fiser_pts = [data[level][m][0] for m in MODELS]
    cemir_pts = [data[level][m][1] for m in MODELS]

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlabel('Success Rate (%)', fontweight='bold', fontsize=13)

    rects1 = ax.barh(y - height/2, fiser_pts, height, label='FISER',
                      color='#3498db', edgecolor='black', linewidth=0.8, zorder=3)
    rects2 = ax.barh(y + height/2, cemir_pts, height, label='CEMIR',
                      color='#e67e22', edgecolor='black', linewidth=0.8, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(MODEL_LABELS, fontweight='bold')
    ax.legend(loc='upper right', frameon=True)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    max_val = max(fiser_pts + cemir_pts) if max(fiser_pts + cemir_pts) > 0 else 10
    ax.set_xlim(0, max_val * 1.25)

    def autolabel_h(rects, values):
        for rect, val in zip(rects, values):
            width = rect.get_width()
            ax.annotate(f'{val:.2f}%',
                        xy=(width, rect.get_y() + rect.get_height() / 2),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=12, fontweight='bold')

    autolabel_h(rects1, fiser_pts)
    autolabel_h(rects2, cemir_pts)

    filepath = os.path.join(FIGURES_DIR, filename)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filepath}")


for level in LEVELS:
    plot_level(level, f"Table3_Level{level}_Updated.png")

print("\nAll four subplots generated. Arrange Level 1+2 as Fig.4 and "
      "Level 3+4 as Fig.5 in your document editor.")
