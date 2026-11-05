"""
AIVA Research Experiment Logger & Provenance Tracker
----------------------------------------------------
Automatically logs experiment execution metadata for strict scientific reproducibility:
- Git commit hash & working tree clean/dirty status
- Dataset version & SHA-256 integrity hash
- Model configuration & hyperparameters
- OS, Python, and hardware execution environment
- Execution timestamp and duration
- Serializes audit receipts to experiments/results/experiment_audit_log.json
"""

import os
import sys
import json
import hashlib
import platform
import subprocess
import datetime
from typing import Dict, Any, Optional


class ExperimentLogger:
    def __init__(self, experiment_name: str, config_path: Optional[str] = None):
        self.experiment_name = experiment_name
        self.config_path = config_path
        self.start_time = datetime.datetime.now()

    @staticmethod
    def get_git_commit_hash() -> str:
        """Retrieves active Git commit hash."""
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            return res.stdout.strip()
        except Exception:
            return "git_commit_unavailable"

    @staticmethod
    def is_git_dirty() -> bool:
        """Checks if there are uncommitted changes."""
        try:
            res = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True
            )
            return len(res.stdout.strip()) > 0
        except Exception:
            return False

    @staticmethod
    def compute_file_sha256(file_path: str) -> str:
        """Computes SHA-256 checksum for dataset version immutability."""
        if not os.path.exists(file_path):
            return "file_not_found"
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def build_provenance_record(
        self,
        dataset_path: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, Any]] = None,
        artifacts: Optional[list] = None
    ) -> Dict[str, Any]:
        end_time = datetime.datetime.now()
        duration_sec = round((end_time - self.start_time).total_seconds(), 2)

        record = {
            "experiment_name": self.experiment_name,
            "config_file": self.config_path,
            "execution_metadata": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration_sec,
                "git_commit": self.get_git_commit_hash(),
                "git_working_tree_dirty": self.is_git_dirty(),
                "python_version": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor()
            },
            "dataset_provenance": {
                "path": dataset_path,
                "sha256": self.compute_file_sha256(dataset_path) if dataset_path else "none"
            },
            "hyperparameters": hyperparameters or {},
            "metrics_reported": metrics or {},
            "output_artifacts": artifacts or []
        }

        # Append to audit log
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
        os.makedirs(log_dir, exist_ok=True)
        audit_file = os.path.join(log_dir, "experiment_audit_log.json")

        existing_logs = []
        if os.path.exists(audit_file):
            try:
                with open(audit_file, "r", encoding="utf-8") as f:
                    existing_logs = json.load(f)
            except Exception:
                existing_logs = []

        existing_logs.append(record)
        with open(audit_file, "w", encoding="utf-8") as f:
            json.dump(existing_logs, f, indent=2)

        return record
