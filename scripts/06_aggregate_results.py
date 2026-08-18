import os
import sys
import shutil

VALID_PIPELINES = ["local_qwen_ablation", "method_comparison"]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_all_experiment_folders(base_path):
    return [name for name in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, name))]

def get_all_runs_in_experiment(exp_path):
    return [name for name in os.listdir(exp_path)
            if os.path.isdir(os.path.join(exp_path, name))]

def aggregate_results(separate_results_path, agg_results_path):
    ensure_dir(agg_results_path)

    experiments = get_all_experiment_folders(separate_results_path)

    for experiment in experiments:
        experiment_path = os.path.join(separate_results_path, experiment)
        run_folders = get_all_runs_in_experiment(experiment_path)

        dest_experiment_path = os.path.join(agg_results_path, experiment)
        ensure_dir(dest_experiment_path)

        run_counter = 0

        for run_folder in sorted(run_folders, key=lambda x: int(x)):
            src_run_path = os.path.join(experiment_path, run_folder)
            dest_run_path = os.path.join(dest_experiment_path, str(run_counter))

            shutil.copytree(src_run_path, dest_run_path)
            run_counter += 1

            print(f"[INFO] Copied {src_run_path} -> {dest_run_path}")

    print(f"\nAggregation complete. Results saved in: {agg_results_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in VALID_PIPELINES:
        print(f"Usage: python {os.path.basename(__file__)} <pipeline>")
        print(f"  <pipeline> must be one of: {VALID_PIPELINES}")
        sys.exit(1)

    pipeline = sys.argv[1]
    separate_results_path = os.path.join("results", pipeline, "separate_results")
    agg_results_path = os.path.join("results", pipeline, "agg_result")

    aggregate_results(separate_results_path, agg_results_path)
