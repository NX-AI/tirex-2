"""Run the GiftEval benchmark with a TiRex2 model.

Usage
-----
::

    pixi run gifteval GIFT_EVAL_STORE MODEL_TYPE [--out RESULTS.csv]

The ``gifteval`` task sets ``PYTHONPATH=src`` so the ``tirex`` package resolves.

The first positional argument is the path to the local GiftEval storage
directory (the one downloaded via ``huggingface-cli download Salesforce/GiftEval``).
The second selects the TiRex2 model to evaluate. ``MODEL_TYPE`` must be
``pretrained`` or ``zero-shot``.
"""

import argparse
import os
from pathlib import Path

DEFAULT_MODEL_PATHS = {
    "pretrained": "NX-AI/TiRex-2-gifteval-pretrain",
    "zero-shot": "NX-AI/TiRex-2-gifteval-zs",
}


def normalize_model_type(value: str) -> str:
    normalized = value.strip().lower().replace("_", "-").replace(" ", "-")
    if normalized == "zeroshot":
        normalized = "zero-shot"
    if normalized not in DEFAULT_MODEL_PATHS:
        choices = ", ".join(DEFAULT_MODEL_PATHS)
        raise argparse.ArgumentTypeError(f"invalid model_type: {value!r} (choose from: {choices})")
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GiftEval benchmark with TiRex2.")
    parser.add_argument("gift_eval_store", help="Path to the local GiftEval storage directory.")
    parser.add_argument(
        "model_type",
        type=normalize_model_type,
        metavar="{pretrained,zero-shot}",
        help="GiftEval model type. Use 'pretrained' or 'zero-shot'.",
    )
    parser.add_argument("--out", default=None, help="Where to write the results CSV (default: ./gifteval_results.csv).")
    parser.add_argument("--device", default="cuda", help="Device to run the model on.")
    parser.add_argument(
        "--eval-mode",
        choices=["univariate", "multivariate"],
        default="multivariate",
        help=(
            "multivariate (default): keep the native [V, T] target and forecast all variates "
            "jointly, exercising the model's variate-mixing path. univariate: split multivariate "
            "datasets into independent channels (the standard GIFT-Eval leaderboard protocol)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import pandas as pd

    store = Path(args.gift_eval_store).expanduser().resolve()
    if not store.is_dir():
        raise FileNotFoundError(f"GiftEval store not found: {store}")
    # The Dataset helper reads the store location from this env var.
    os.environ["GIFT_EVAL"] = str(store)

    # Import after GIFT_EVAL is set (the Dataset helper reads it at import time).
    from gift_eval_utils import TiRexGiftEvalWrapper, evaluate_dataset, gift_eval_dataset_iter

    from tirex2 import load_model

    model_path = DEFAULT_MODEL_PATHS[args.model_type]
    load_device = "cuda" if args.device.startswith("cuda") else args.device
    print(f"Loading {args.model_type} model from {model_path}")
    model = load_model(model_path, device=load_device)
    model.model.to(args.device)

    eval_multivariate = args.eval_mode == "multivariate"
    print(f"Evaluation mode: {args.eval_mode}")
    wrapped_model = TiRexGiftEvalWrapper(model, eval_multivariate=eval_multivariate)

    results = []
    for task in gift_eval_dataset_iter():
        task_result = evaluate_dataset(wrapped_model, eval_multivariate=eval_multivariate, **task)
        results.append(task_result)
        print(task_result)

    results = pd.DataFrame(results)
    out_path = Path(args.out) if args.out else Path.cwd() / "gifteval_results.csv"
    results.to_csv(out_path, index=False)
    print(f"\nWrote {len(results)} results to {out_path}")


if __name__ == "__main__":
    main()
