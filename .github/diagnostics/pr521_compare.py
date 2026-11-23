"""Compare immutable base/head CI outcomes on one native runner and environment."""

from __future__ import annotations

import argparse
import contextlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET

BASE = "444e548ee39eed4aaadbde64416fd495f6f109ab"
HEAD = "57d6b665c0ab4c8c03d3eef534d61fecda7b076b"
CONFIG = Path(__file__).resolve().parent
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_CORETYPE",
    "LOKY_MAX_CPU_COUNT",
)


def _write(path, value):
    """Save machine-readable observations with real newlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _normalized(name):
    """Normalize Python distribution names for exact pin comparison."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _pins():
    """Read the 44 versions from the original failed CI job."""
    return {
        _normalized(k): v
        for k, v in (
            line.strip().split("==", 1)
            for line in (CONFIG / "pr521_minimum_pins.txt").read_text().splitlines()
            if line.strip()
        )
    }


def _packages():
    """Return this environment's complete distribution inventory."""
    return {
        _normalized(d.metadata["Name"]): d.version
        for d in importlib.metadata.distributions()
    }


def inventory(source, output):
    """Assert exact software/source inputs and capture native CPU/BLAS context."""
    import cvxpy
    import joblib
    import numpy as np
    import scipy
    import skfolio
    from threadpoolctl import threadpool_info

    expected = _pins() | {"skfolio": "1.14.4"}
    assert _packages() == expected, (_packages(), expected)
    assert sys.version_info[:3] == (3, 10, 22), sys.version
    assert platform.machine() == "x86_64", platform.machine()
    release = dict(
        line.split("=", 1)
        for line in Path("/etc/os-release").read_text().splitlines()
        if "=" in line
    )
    assert release["VERSION_ID"].strip('"') == "26.04", release
    assert Path(skfolio.__file__).resolve().is_relative_to(source / "src"), (
        skfolio.__file__
    )
    cpu = subprocess.run(
        ["lscpu", "--json"], capture_output=True, text=True, check=True
    )
    config = io.StringIO()
    with contextlib.redirect_stdout(config):
        np.show_config()
        scipy.show_config()
    _write(
        output,
        {
            "python": sys.version,
            "machine": platform.machine(),
            "os_release": release,
            "cpu": json.loads(cpu.stdout),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "joblib_cpu_count": joblib.cpu_count(),
            "thread_environment": {k: os.environ.get(k) for k in THREAD_KEYS},
            "threadpools": threadpool_info(),
            "packages": _packages(),
            "source": skfolio.__file__,
            "cvxpy_version": cvxpy.__version__,
            "numpy_scipy_build_config": config.getvalue(),
        },
    )


def _output(values):
    """Expose small result counts to subsequent workflow step names."""
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            for key, value in values.items():
                f.write(f"{key}={value}\n")


def run(label, source, commit, results):
    """Run the unchanged full test suite, preserving native failures as data."""
    phase = results / label
    phase.mkdir(parents=True, exist_ok=False)
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    assert actual == commit, (actual, commit)
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join((str(source / "src"), str(CONFIG))),
        VIRTUAL_ENV=os.environ["PAIR_ENV"],
        UV_PROJECT_ENVIRONMENT=os.environ["PAIR_ENV"],
        UV_PYTHON=sys.executable,
        UV_PYTHON_DOWNLOADS="never",
        UV_OFFLINE="1",
        PR521_PHASE_RESULTS=str(phase),
    )
    runtime = results.parent / "pr521-runtime" / label
    env.update(
        SKFOLIO_DATA=str(runtime / "datasets"),
        MPLCONFIGDIR=str(runtime / "matplotlib"),
        NUMBA_CACHE_DIR=str(runtime / "numba"),
    )
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "inventory",
            "--source",
            str(source),
            "--output",
            str(phase / "environment.json"),
        ],
        cwd=source,
        env=env,
        check=True,
    )
    command = [
        "uv",
        "run",
        "--no-sync",
        "pytest",
        "-q",
        "tests",
        "-p",
        "pr521_failure_capture",
        "--junitxml=" + str(phase / "junit.xml"),
    ]
    _write(
        phase / "command.json",
        {
            "commit": commit,
            "cwd": str(source),
            "argv": command,
            "diagnostics_only": True,
            "assertions_modified": False,
        },
    )
    start = time.monotonic()
    stream_error = []
    with (phase / "pytest.log").open("w") as log:
        proc = subprocess.Popen(
            command,
            cwd=source,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
            start_new_session=True,
        )

        def pump():
            """Tee native output without touching numerical state in pytest."""
            try:
                for line in proc.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
            except Exception as exc:
                stream_error.append(repr(exc))

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        try:
            code = proc.wait(timeout=720)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = 124
        reader.join(timeout=30)
    failures = sorted({
        data["nodeid"]
        for p in (phase / "diagnostics").glob("*-failure.json")
        if (data := json.loads(p.read_text()))["when"] == "call"
    })
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    if (phase / "junit.xml").exists():
        root = ET.parse(phase / "junit.xml").getroot()
        for suite in root.iter("testsuite"):
            for key in counts:
                counts[key] += int(suite.get(key, "0"))
    if code in (0, 1) and (not counts["tests"] or len(failures) != counts["failures"]):
        raise RuntimeError("Missing/inconsistent native test result data")
    counts["passed"] = (
        counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    )
    before = json.loads((phase / "environment.json").read_text())["packages"]
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "inventory",
            "--source",
            str(source),
            "--output",
            str(phase / "environment-after.json"),
        ],
        cwd=source,
        env=env,
        check=True,
    )
    assert (
        before == json.loads((phase / "environment-after.json").read_text())["packages"]
    )
    assert not subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True
    ).strip(), "Tracked source changed"
    result = {
        "label": label,
        "commit": commit,
        "native_exit_code": code,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "counts": counts,
        "failure_nodes": failures,
        "stream_errors": stream_error,
        "source_unchanged": True,
        "package_inventory_unchanged": True,
    }
    _write(phase / "result.json", result)
    _output(
        {"native_exit": code, "failed": counts["failures"], "passed": counts["passed"]}
    )
    print("PHASE_RESULT " + json.dumps(result), flush=True)
    if code not in (0, 1) or stream_error or reader.is_alive() or counts["errors"]:
        raise SystemExit("Incomplete diagnostic phase; native test data retained")


def decide(results):
    """Repeat in reverse order only when the first failure sets disagree."""
    base = json.loads((results / "base-1/result.json").read_text())
    head = json.loads((results / "pr-1/result.json").read_text())
    repeat = base["failure_nodes"] != head["failure_nodes"]
    _write(
        results / "repeat-decision.json",
        {
            "repeat": repeat,
            "reason": "Different failure sets require a stability check"
            if repeat
            else "Same failure sets; no duplicate full suites needed",
        },
    )
    _output({"repeat": str(repeat).lower()})
    print("Reverse-order repeat:", repeat)


def summarize(results):
    """Make complete versus partial evidence explicit, regardless of test outcome."""
    phases = [json.loads(p.read_text()) for p in results.glob("*/result.json")]
    by_label = {p["label"]: p for p in phases}
    complete = "base-1" in by_label and "pr-1" in by_label
    decision = results / "repeat-decision.json"
    repeated = decision.exists() and json.loads(decision.read_text())["repeat"]
    complete = (
        complete
        and decision.exists()
        and (not repeated or all(x in by_label for x in ("pr-2", "base-2")))
    )
    sets = {p["label"]: p["failure_nodes"] for p in phases}
    if not complete:
        finding = "Incomplete comparison; inspect setup/native phase errors."
    elif not repeated and not sets["base-1"]:
        finding = "Both full suites passed. Original failures not reproduced; cause unresolved."
    elif not repeated:
        finding = "The same failures occur on accepted base and PR head."
    elif sets.get("base-1") == sets.get("base-2") and sets.get("pr-1") == sets.get(
        "pr-2"
    ):
        finding = "Failure sets differ consistently between source/test sets; further isolation required."
    else:
        finding = "Failure sets vary within an unchanged commit; runtime/numerical variation observed."
    summary = {
        "base_commit": BASE,
        "pr_commit": HEAD,
        "comparison_complete": complete,
        "reverse_repeat_requested": repeated,
        "finding": finding,
        "phases": phases,
        "note": "This diagnostic completion is not an upstream PR CI pass or maintainer approval.",
    }
    _write(results / "summary.json", summary)
    lines = [
        "# PR 521 minimum-dependency paired comparison",
        "",
        finding,
        "",
        "Diagnostic completion does not mean the native suites passed.",
        "",
        "| Phase | Commit | Native exit | Passed | Failed | Skipped | Seconds |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("base-1", "pr-1", "pr-2", "base-2"):
        if name not in by_label:
            continue
        p = by_label[name]
        c = p["counts"]
        lines.append(
            f"| {name} | `{p['commit'][:10]}` | {p['native_exit_code']} | {c['passed']} | {c['failures']} | {c['skipped']} | {p['elapsed_seconds']} |"
        )
    for name in ("base-1", "pr-1", "pr-2", "base-2"):
        p = by_label.get(name)
        if p and p["failure_nodes"]:
            lines += [
                "",
                f"**{name} failures**",
                "",
                *["- `" + n + "`" for n in p["failure_nodes"]],
                "",
            ]
    text = "\n".join(lines) + "\n"
    (results / "summary.md").write_text(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)
    print(json.dumps(summary), flush=True)


def main():
    """Expose small commands used by the fork-only workflow."""
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("inventory")
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("run")
    p.add_argument("--label", required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--commit", required=True)
    for name in ("decide", "summarize"):
        commands.add_parser(name)
    args = parser.parse_args()
    results = Path(os.environ.get("PAIR_RESULTS", "."))
    if args.command == "inventory":
        inventory(args.source.resolve(), args.output)
    elif args.command == "run":
        run(args.label, args.source.resolve(), args.commit, results)
    elif args.command == "decide":
        decide(results)
    else:
        summarize(results)


if __name__ == "__main__":
    main()
