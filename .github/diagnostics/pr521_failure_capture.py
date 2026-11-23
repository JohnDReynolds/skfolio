"""Record failure values and worker context without changing native assertions."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

import pytest


def _directory(config):
    """Return this phase's isolated diagnostics directory."""
    return Path(os.environ["PR521_PHASE_RESULTS"]) / "diagnostics"


def _safe(value):
    """Encode exact scalar/array values without global print-option changes."""
    np = sys.modules.get("numpy")
    if np is not None:
        if isinstance(value, np.ndarray):
            if value.size <= 256:
                return _safe(value.tolist())
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            }
        if isinstance(value, np.generic):
            return _safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    return repr(value)


def pytest_collection_finish(session):
    """Record already-loaded math libraries and collection order per worker."""
    directory = _directory(session.config)
    directory.mkdir(parents=True, exist_ok=True)
    worker = getattr(session.config, "workerinput", {}).get("workerid", "controller")
    ctl = sys.modules.get("threadpoolctl")
    np = sys.modules.get("numpy")
    data = {
        "worker": worker,
        "pid": os.getpid(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": {
            k: os.environ.get(k)
            for k in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_CORETYPE",
                "LOKY_MAX_CPU_COUNT",
            )
        },
        "threadpools": ctl.threadpool_info() if ctl is not None else [],
        "numpy_loaded": np is not None,
    }
    (directory / f"{worker}-context.json").write_text(json.dumps(_safe(data), indent=2))
    if worker in ("gw0", "controller"):
        (directory / f"{worker}-collection.txt").write_text(
            "\n".join(item.nodeid for item in session.items) + "\n"
        )


def pytest_runtest_logstart(nodeid, location):
    """Retain actual test order to distinguish scheduling from code effects."""
    if "PYTEST_XDIST_WORKER" not in os.environ:
        return
    directory = Path(os.environ["PR521_PHASE_RESULTS"]) / "diagnostics"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"{os.environ['PYTEST_XDIST_WORKER']}-order.txt").open("a") as f:
        f.write(nodeid + "\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Observe failed native assertions; preserve their exception and outcome."""
    outcome = yield
    report = outcome.get_result()
    if not report.failed or call.excinfo is None:
        return
    directory = _directory(item.config)
    directory.mkdir(parents=True, exist_ok=True)
    worker = getattr(item.config, "workerinput", {}).get("workerid", "controller")
    key = hashlib.sha256((item.nodeid + call.when).encode()).hexdigest()[:16]
    data = {
        "nodeid": item.nodeid,
        "when": call.when,
        "worker": worker,
        "exception_type": call.excinfo.type.__name__,
        "exception": str(call.excinfo.value),
        "frames": [],
    }
    np = sys.modules.get("numpy")
    arrays = {}
    try:
        for frame in call.excinfo.traceback:
            local = frame.frame.f_locals
            details = {
                "path": str(frame.path),
                "line": frame.lineno + 1,
                "function": frame.name,
            }
            if "p" in local and "p_expected" in local and np is not None:
                p, expected = local["p"], local["p_expected"]
                if hasattr(p, "weights") and hasattr(expected, "weights"):
                    actual_weights = np.asarray(p.weights)
                    expected_weights = np.asarray(expected.weights)
                    difference = actual_weights - expected_weights
                    details["weights"] = _safe(actual_weights)
                    details["expected_weights"] = _safe(expected_weights)
                    details["absolute_difference"] = _safe(np.abs(difference))
                    details["max_absolute_difference"] = float(
                        np.max(np.abs(difference))
                    )
                    details["assets"] = _safe(getattr(p, "assets", None))
                    details["observations"] = _safe(getattr(p, "observations", None))
                    for name in ("previous_weights", "returns"):
                        if hasattr(p, name):
                            details[name] = _safe(np.asarray(getattr(p, name)))
                            details["expected_" + name] = _safe(
                                np.asarray(getattr(expected, name))
                            )
                    arrays.update(
                        weights=actual_weights, expected_weights=expected_weights
                    )
                    mpp = local.get("mpp")
                    if mpp is not None:
                        details["estimator_name"] = getattr(mpp, "name", None)
                        details["period"] = next(
                            (i for i, x in enumerate(mpp.portfolios) if x is p), None
                        )
            if "sol" in local:
                sol = local["sol"]
                details["solver_result"] = {
                    name: _safe(getattr(sol, name, None))
                    for name in (
                        "success",
                        "status",
                        "message",
                        "nit",
                        "nfev",
                        "fun",
                        "x",
                        "jac",
                    )
                }
                if np is not None:
                    for name in ("a", "b"):
                        if name in local:
                            arrays["solver_" + name] = np.asarray(local[name])
                    model = local.get("self")
                    if model is not None:
                        for name in ("_returns", "_prior_sample_weight"):
                            if hasattr(model, name):
                                arrays[name] = np.asarray(getattr(model, name))
            if "corr" in local and "corr_prior" in local:
                details["correlation"] = _safe(local["corr"])
                details["prior_correlation"] = _safe(local["corr_prior"])
            if len(details) > 3:
                data["frames"].append(details)
        ctl = sys.modules.get("threadpoolctl")
        data["threadpools_at_failure"] = (
            ctl.threadpool_info() if ctl is not None else []
        )
        if arrays and np is not None:
            np.savez_compressed(directory / f"{worker}-{key}-arrays.npz", **arrays)
    except Exception:
        data["capture_error"] = traceback.format_exc()
    (directory / f"{worker}-{key}-failure.json").write_text(
        json.dumps(_safe(data), indent=2, allow_nan=False)
    )
