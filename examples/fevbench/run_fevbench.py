"""Run the fev-bench benchmark with TiRex2.

Usage
-----
::

    pixi run --platform linux-64-cuda fevbench [--tasks tasks.yaml] [--out RESULTS.csv]

Tasks are loaded from their configured paths in the YAML, typically the
HuggingFace Hub (``autogluon/fev_datasets``). The model is always loaded from
``NX-AI/TiRex-2-fevbench``.

To run against a local dataset snapshot (offline), point HuggingFace's datasets
cache at it rather than passing a path here. The snapshot must use the standard
cache layout (``<cache>/autogluon___fev_datasets/<config>/...``)::

    HF_DATASETS_CACHE=/path/to/snapshot HF_HUB_OFFLINE=1 pixi run --platform linux-64-cuda fevbench

The benchmark itself is described by a YAML file (``--tasks``, default
``tasks.yaml`` next to this script) listing the fev tasks to run. Each evaluation
task is forecast through :class:`submission.TiRex2Model` and scored with
``fev.Task.evaluation_summary``; the per-task summaries are written to a CSV.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

DEFAULT_MODEL_PATH = "NX-AI/TiRex-2-fevbench"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fev-bench benchmark with TiRex2.")
    parser.add_argument(
        "--tasks",
        default=None,
        help="Path to the benchmark tasks YAML (default: tasks.yaml next to this script).",
    )
    parser.add_argument("--out", default=None, help="Where to write the results CSV (default: ./fevbench_results.csv).")
    parser.add_argument("--device", default="cuda", help="Device to run the model on.")
    parser.add_argument("--batch_size", type=int, default=512, help="Forecast batch size.")
    parser.add_argument("--max_tasks", type=int, default=None, help="Run only the first N tasks from the YAML file.")
    parser.add_argument(
        "--as_univariate",
        action="store_true",
        help="Predict each target independently, ignoring covariates.",
    )
    parser.add_argument("--model_name", default=None, help="Model name recorded in the results.")
    return parser.parse_args()


def load_benchmark(tasks_yaml: Path, max_tasks: int | None = None):
    """Build a :class:`fev.Benchmark` from a tasks YAML."""
    import fev

    with open(tasks_yaml, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    task_configs = config["tasks"] if max_tasks is None else config["tasks"][:max_tasks]
    tasks = [fev.Task(**task_config) for task_config in task_configs]
    return fev.Benchmark(tasks=tasks)


def main() -> None:
    args = parse_args()

    tasks_yaml = Path(args.tasks) if args.tasks else Path(__file__).parent / "tasks.yaml"
    if not tasks_yaml.is_file():
        raise FileNotFoundError(f"Tasks YAML not found: {tasks_yaml}")

    from submission import TiRex2Model

    benchmark = load_benchmark(tasks_yaml, max_tasks=args.max_tasks)
    model = TiRex2Model(
        model_path=DEFAULT_MODEL_PATH,
        batch_size=args.batch_size,
        device=args.device,
        as_univariate=args.as_univariate,
    )

    summaries = []
    for task in benchmark.tasks:
        print(f"Processing task: {task.task_name}")
        predictions_per_window = model.fit_predict(task)
        summary = task.evaluation_summary(
            predictions_per_window,
            model_name=model.model_name if args.model_name is None else args.model_name,
            training_time_s=model.training_time,
            inference_time_s=model.inference_time,
            trained_on_this_dataset=task.dataset_config in model.trained_on_datasets,
        )
        summaries.append(summary)
        print(summary)

    results = pd.DataFrame(summaries)
    out_path = Path(args.out) if args.out else Path.cwd() / "fevbench_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nWrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
