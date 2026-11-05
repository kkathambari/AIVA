"""
AIVA Unified Experiment Runner & Orchestration Entrypoint
---------------------------------------------------------
Executes reproducible benchmarks from standardized YAML experiment specifications.

Usage:
    python -m experiments.run --config experiments/configs/ucs.yaml
    python experiments/run.py --config experiments/configs/kg.yaml
    python experiments/run.py --all
"""

import os
import sys
import argparse
import subprocess
import yaml
from typing import Dict, Any

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AIVA_backend")))

from experiments.tracking.experiment_logger import ExperimentLogger


CONFIG_REGISTRY = {
    "kg": "experiments/configs/kg.yaml",
    "rag": "experiments/configs/rag.yaml",
    "ucs": "experiments/configs/ucs.yaml",
    "grading": "experiments/configs/grading.yaml",
    "difficulty": "experiments/configs/difficulty.yaml"
}


def run_experiment_from_config(config_path: str):
    print("=" * 80)
    print(f"AIVA EXPERIMENT RUNNER: Executing {config_path}")
    print("=" * 80)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    exp_name = cfg.get("experiment", {}).get("name", os.path.basename(config_path))
    eval_cfg = cfg.get("evaluation", {})
    script_rel_path = eval_cfg.get("script")

    if not script_rel_path:
        raise ValueError(f"No evaluation script designated in {config_path}")

    script_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", script_rel_path))
    if not os.path.exists(script_abs_path):
        raise FileNotFoundError(f"Evaluation script not found: {script_abs_path}")

    logger = ExperimentLogger(experiment_name=exp_name, config_path=config_path)

    # Execute experiment script via current Python interpreter
    python_exe = sys.executable
    print(f"Launching: {python_exe} {script_rel_path}")
    cmd = [python_exe, script_abs_path]
    res = subprocess.run(cmd, capture_output=True, text=True)

    print(res.stdout)
    if res.stderr:
        print("STDERR:")
        print(res.stderr)

    if res.returncode != 0:
        print(f"[FAILED] Experiment exited with code {res.returncode}")
        sys.exit(res.returncode)

    # Read output artifact if specified
    out_artifact = eval_cfg.get("output_artifact") or eval_cfg.get("output_json")
    metrics_data = {}
    if out_artifact:
        art_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", out_artifact))
        if os.path.exists(art_path):
            try:
                import json
                with open(art_path, "r", encoding="utf-8") as f:
                    metrics_data = json.load(f)
            except Exception:
                pass

    dataset_path = cfg.get("dataset", {}).get("path") or cfg.get("dataset", {}).get("knowledge_graph_dataset")

    # Record provenance receipt
    receipt = logger.build_provenance_record(
        dataset_path=dataset_path,
        hyperparameters=cfg.get("simulation") or cfg.get("pipeline") or cfg.get("models"),
        metrics=metrics_data,
        artifacts=[out_artifact] if out_artifact else []
    )

    print("\n[SUCCESS] Experiment completed and logged.")
    print(f"Audit Receipt recorded in: experiments/results/experiment_audit_log.json")
    print(f"Git Commit: {receipt['execution_metadata']['git_commit']}")
    print(f"Execution Duration: {receipt['execution_metadata']['duration_seconds']}s")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="AIVA Unified Experiment Runner")
    parser.add_argument("--config", type=str, help="Path to experiment YAML configuration file (e.g., experiments/configs/ucs.yaml)")
    parser.add_argument("--all", action="store_true", help="Execute all standardized benchmark experiments sequentially")
    parser.add_argument("--list", action="store_true", help="List registered experiment configurations")

    args = parser.parse_args()

    if args.list:
        print("Available Experiment Configurations:")
        for key, path in CONFIG_REGISTRY.items():
            print(f"  {key:<12}: {path}")
        return

    if args.all:
        print("Executing all research experiments in sequence...")
        for key, path in CONFIG_REGISTRY.items():
            run_experiment_from_config(path)
        print("\nAll experiments successfully completed.")
        return

    if args.config:
        run_experiment_from_config(args.config)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
