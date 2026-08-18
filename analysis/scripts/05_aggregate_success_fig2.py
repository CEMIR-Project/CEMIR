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

os.makedirs(FIGURES_DIR, exist_ok=True)


def point_estimate_sr(df, model, method):
    sub = df[(df["model"] == model) & (df["method"] == method) &
             (df["repetition"].isin(VALID_REPS))]
    return sub[METRIC].mean() * 100


df = pd.read_csv(MAIN_FILE)
df["method"] = df["method"].astype(str).str.strip()
df["model"] = df["model"].astype(str).str.strip()

fiser_sr, cemir_sr = [], []

for model in MODELS:
    f_pt = point_estimate_sr(df, model, METHOD_FISER)
    c_pt = point_estimate_sr(df, model, METHOD_CEMIR)

    fiser_sr.append(f_pt)
    cemir_sr.append(c_pt)

    print(f"{model}: FISER={f_pt:.2f}%  CEMIR={c_pt:.2f}%")

x = np.arange(len(MODELS))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 7))
ax.set_axisbelow(True)
ax.yaxis.grid(True, linestyle='--', alpha=0.6, zorder=0)

rects1 = ax.bar(x - width/2, fiser_sr, width, label='FISER', color='#3498db',
                 edgecolor='black', linewidth=0.8, alpha=0.9, zorder=3)
rects2 = ax.bar(x + width/2, cemir_sr, width, label='CEMIR', color='#e67e22',
                 edgecolor='black', linewidth=0.8, alpha=0.9, zorder=3)

ax.set_ylabel('Success Rate (%)', fontweight='bold', fontsize=12)
ax.set_title('Aggregate Success Rates of FISER vs. CEMIR Across Different Models',
             pad=20, fontweight='bold', fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(MODEL_LABELS, fontweight='bold')
ax.set_ylim(0, 35)
ax.legend(frameon=True, facecolor='white', edgecolor='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


def autolabel(rects, values):
    for rect, val in zip(rects, values):
        height = rect.get_height()
        ax.annotate(f'{val:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

autolabel(rects1, fiser_sr)
autolabel(rects2, cemir_sr)

plt.tight_layout()
out_path = os.path.join(FIGURES_DIR, "Fig2_Aggregate_Success_Rate_Updated.png")
plt.savefig(out_path, dpi=300, bbox_inches='tight')

print(f"\nSaved: {out_path}")
