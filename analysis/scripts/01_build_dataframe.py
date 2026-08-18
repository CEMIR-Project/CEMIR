import os
import json
import re
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

ABLATION_RESULTS_DIR = os.path.join(REPO_ROOT, "results", "local_qwen_ablation", "agg_result")
METHOD_COMPARISON_RESULTS_DIR = os.path.join(REPO_ROOT, "results", "method_comparison", "agg_result")

ABLATION_OUT_CSV = os.path.join(DATA_DIR, "ablation_customized_new_Qwen.csv")
METHOD_COMPARISON_OUT_CSV = os.path.join(DATA_DIR, "final_whole_summary.csv")

ABLATION_INCLUDED_EXPERIMENTS = [
    "title_enhanced",
    "title_image",
]

ABLATION_EXPERIMENT_LABELS = {
    "title_enhanced": "Raw Text + Enhanced Text",
    "title_image": "Raw Text + Image",
}

ABLATION_BASELINE_FOLDER = "FISER_Qwen_3b"
ABLATION_BASELINE_MODEL = "Qwen_3b"
ABLATION_BASELINE_MODALITY = "FISER"


def extract_object_and_id(line):
    match = re.search(r'give(?:s)?(?: the)?(?: [a-z]+)* ([a-zA-Z_]+) (\d+)', line)
    if match:
        return match.group(1).lower(), match.group(2)
    match = re.search(r'give(?:s)?(?: the)?(?: [a-z]+)* ([a-zA-Z_]+)\s+to human', line)
    if match:
        return match.group(1).lower(), None
    return None, None


def evaluate_episode(data):
    assistant_actions = data.get("assistant", "").strip().split("Actions:")
    demo_actions = data.get("demo_answer", "").strip().splitlines()

    exact_acc = 0

    if not assistant_actions or not demo_actions or len(assistant_actions) < 2:
        return exact_acc

    assistant_actions = assistant_actions[1].splitlines()

    assistant_line = None
    for line in assistant_actions:
        if "give" in line.lower() and "to human" in line.lower():
            assistant_line = line.lower()
            break

    demo_line = demo_actions[-1].lower()
    a_obj, a_id = extract_object_and_id(assistant_line) if assistant_line else (None, None)
    d_obj, d_id = extract_object_and_id(demo_line)

    if a_obj and d_obj and a_obj == d_obj:
        if a_id == d_id or d_id is None:
            exact_acc = 1

    return exact_acc


def extract_level(filename):
    match = re.search(r'-(\d+)\.json$', filename)
    if match:
        return int(match.group(1))
    return None


def collect_rows(results_dir, experiment_folder, model, modality):
    rows = []
    experiment_path = os.path.join(results_dir, experiment_folder)
    if not os.path.isdir(experiment_path):
        return rows

    for run_folder in os.listdir(experiment_path):
        run_path = os.path.join(experiment_path, run_folder)
        if not os.path.isdir(run_path):
            continue

        try:
            repetition = int(run_folder)
        except ValueError:
            repetition = 0

        for filename in os.listdir(run_path):
            if not filename.endswith(".json"):
                continue

            level = extract_level(filename)
            if level is None:
                continue

            filepath = os.path.join(run_path, filename)
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
            except Exception:
                continue

            episode_id = filename.replace(".json", "")
            exact_acc = evaluate_episode(data)

            rows.append({
                "model": model,
                "modality": modality,
                "level": level,
                "episode_id": episode_id,
                "repetition": repetition,
                "exact_acc": exact_acc,
                "run_folder": run_folder,
                "filename": filename,
            })

    return rows


def build_ablation_dataframe():
    all_rows = []

    for experiment_folder in ABLATION_INCLUDED_EXPERIMENTS:
        all_rows.extend(
            collect_rows(
                ABLATION_RESULTS_DIR,
                experiment_folder,
                "Qwen_Omni",
                ABLATION_EXPERIMENT_LABELS[experiment_folder],
            )
        )

    all_rows.extend(
        collect_rows(
            METHOD_COMPARISON_RESULTS_DIR,
            ABLATION_BASELINE_FOLDER,
            ABLATION_BASELINE_MODEL,
            ABLATION_BASELINE_MODALITY,
        )
    )

    df = pd.DataFrame(all_rows)
    df.to_csv(ABLATION_OUT_CSV, index=False)

    print(f"Saved: {ABLATION_OUT_CSV}")
    print(f"Total episodes: {len(df)}")
    if len(df) > 0:
        print(f"Columns: {list(df.columns)}")


def parse_experiment_folder(folder_name):
    for method in ["CEMIR", "FISER"]:
        if folder_name.startswith(method + "_"):
            model = folder_name[len(method) + 1:]
            return method, model, False

    return None, None, True


def build_method_comparison_dataframe():
    all_episodes = []

    if not os.path.isdir(METHOD_COMPARISON_RESULTS_DIR):
        print(f"[WARNING] No episodes found under {METHOD_COMPARISON_RESULTS_DIR}")
        return

    for experiment_folder in os.listdir(METHOD_COMPARISON_RESULTS_DIR):
        experiment_path = os.path.join(METHOD_COMPARISON_RESULTS_DIR, experiment_folder)
        if not os.path.isdir(experiment_path):
            continue

        method, model, skip = parse_experiment_folder(experiment_folder)

        if skip:
            print(f"[SKIP] Could not parse method/model from folder name: {experiment_folder}")
            continue

        for run_folder in os.listdir(experiment_path):
            run_path = os.path.join(experiment_path, run_folder)
            if not os.path.isdir(run_path):
                continue

            try:
                repetition = int(run_folder)
            except ValueError:
                repetition = 0

            for filename in os.listdir(run_path):
                if not filename.endswith(".json"):
                    continue

                level_match = re.search(r'level[_-](\d+)', filename)
                level = int(level_match.group(1)) if level_match else None

                if level is None:
                    for pattern in [r'-(\d+)\.json$', r'_(\d+)\.json$']:
                        match = re.search(pattern, filename)
                        if match:
                            try:
                                level = int(match.group(1))
                                break
                            except ValueError:
                                pass

                episode_id = filename.replace(".json", "")
                filepath = os.path.join(run_path, filename)

                try:
                    with open(filepath, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue

                exact_acc = evaluate_episode(data)

                all_episodes.append({
                    'model': model,
                    'method': method,
                    'level': level,
                    'episode_id': episode_id,
                    'repetition': repetition,
                    'exact_acc': exact_acc,
                    'experiment_folder': experiment_folder,
                    'run_folder': run_folder,
                    'filename': filename
                })

    df = pd.DataFrame(all_episodes)

    if len(df) == 0:
        print(f"[WARNING] No episodes found under {METHOD_COMPARISON_RESULTS_DIR}")
        return

    df.to_csv(METHOD_COMPARISON_OUT_CSV, index=False)
    print(f"Saved: {METHOD_COMPARISON_OUT_CSV}")
    print(f"Total episodes: {len(df)}")


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=== Building local Qwen ablation dataframe ===")
    build_ablation_dataframe()

    print("\n=== Building method comparison dataframe ===")
    build_method_comparison_dataframe()
