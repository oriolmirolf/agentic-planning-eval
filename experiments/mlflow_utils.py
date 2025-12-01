# /Oriol-TFM/experiments/mlflow_utils.py
from __future__ import annotations

import os
import os.path
from contextlib import contextmanager
from typing import Any

import mlflow  # type: ignore

# Optional deps for dataset logging
try:
    import pandas as _pd  # type: ignore
except Exception:  # pragma: no cover - optional
    _pd = None

_MLFLOW_CONFIGURED = False


def _enabled() -> bool:
    """Return True if MLflow is available and not explicitly disabled."""
    if mlflow is None:
        return False
    if os.getenv("MLFLOW_DISABLE", "0") == "1":
        return False
    return True


def _ensure_setup() -> None:
    """Configure tracking URI and experiment name once per process."""
    global _MLFLOW_CONFIGURED
    if not _enabled() or _MLFLOW_CONFIGURED:
        return

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)

    # You can override this per-shell:
    #   export MLFLOW_EXPERIMENT="Oriol-TFM-Blocks"
    exp_name = (
        os.getenv("MLFLOW_EXPERIMENT")
        or os.getenv("MLFLOW_EXPERIMENT_NAME")
        or "Oriol-TFM"
    )
    mlflow.set_experiment(exp_name)
    _MLFLOW_CONFIGURED = True


@contextmanager
def mlflow_run(
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
):
    """
    Context manager that starts an MLflow run if MLflow is available.

    Usage:
        with mlflow_run("run-name", tags={"domain": "blocks"}) as run:
            if run is not None:
                log_params(...)
                log_metrics(...)
                log_artifacts(...)
    """
    if not _enabled():
        # No-MLflow mode: yield None so callers can branch.
        yield None
        return

    _ensure_setup()
    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        yield run


def log_params(params: dict[str, Any]) -> None:
    """Log params, coercing everything into MLflow-friendly types."""
    if not _enabled():
        return
    flat: dict[str, Any] = {}
    for k, v in params.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = str(v)
    if flat:
        mlflow.log_params(flat)


def log_metrics(metrics: dict[str, Any]) -> None:
    """Log metrics (floats); None values are ignored."""
    if not _enabled():
        return
    clean: dict[str, float] = {}
    for k, v in metrics.items():
        if v is None:
            continue
        try:
            clean[k] = float(v)
        except (TypeError, ValueError):
            continue
    if clean:
        mlflow.log_metrics(clean)


def log_artifacts(path: str, artifact_path: str | None = None) -> None:
    """
    Log a directory (log_artifacts) or a single file (log_artifact) under
    optional artifact_path.
    """
    if not _enabled():
        return
    p = os.path.abspath(path)
    if os.path.isdir(p):
        mlflow.log_artifacts(p, artifact_path=artifact_path)
    elif os.path.isfile(p):
        mlflow.log_artifact(p, artifact_path=artifact_path)


def log_dataset_for_domain(
    *,
    domain: str,
    domain_path: str,
) -> None:
    """
    Log an MLflow Dataset representing the *benchmark domain* (not a single
    problem instance).

    This is what populates the **Dataset** column in the UI.
    Per-run variation (problem index, etc.) is kept as params/tags.
    """
    if not _enabled():
        return
    if _pd is None:
        # pandas not installed -> silently skip dataset logging
        return

    data_mod = getattr(mlflow, "data", None)
    if data_mod is None or not hasattr(data_mod, "from_pandas"):
        # Older MLflow without dataset API
        return

    domain_path_abs = os.path.abspath(domain_path)
    base_dir = os.path.dirname(domain_path_abs)

    problems_root = os.path.join(base_dir, "problems_pddl")
    if not os.path.isdir(problems_root):
        problems_root = None

    prompts_json = os.path.join(base_dir, "prompts.json")
    if not os.path.isfile(prompts_json):
        prompts_json = None

    df = _pd.DataFrame(
        [
            {
                "domain": domain,
                "domain_path": domain_path_abs,
                "problems_root": problems_root,
                "prompts_json": prompts_json,
            }
        ]
    )

    # Name shown in the "Dataset" column – use the domain only.
    ds_name = domain
    dataset = data_mod.from_pandas(
        df,
        source=base_dir,
        name=ds_name,
    )
    # context="evaluation" is semantically correct here
    mlflow.log_input(dataset, context="evaluation")
