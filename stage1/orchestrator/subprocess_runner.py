"""Launches each model's long-lived inference subprocess (direct interpreter path, never
`conda activate`/`conda run`) and schedules them across the shared GPU cluster.

Per-sample timeouts are handled INSIDE each model subprocess via SIGALRM (bench_common /
_model_infer_base.sample_alarm). The watchdog here is a coarser safety net for a whole
subprocess actually hanging (e.g. a CUDA driver deadlock) in a way a self-interrupt can't catch:
if a model's results.jsonl hasn't grown in `stall_minutes`, the subprocess is killed. Because
run_model_loop resumes from already-written sample_ids, simply re-invoking this function with
the same manifest/results paths picks up where a killed run left off -- that is the recovery
procedure for a stalled model, not automatic retry-in-place.
"""
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from bench_common import io_utils
from orchestrator.config import ModelSpec
from orchestrator.gpu_selector import select_idle_gpu

logger = logging.getLogger(__name__)

POLL_INTERVAL_SECONDS = 20


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class RunResult:
    model_name: str
    exit_code: int
    started_at: str
    ended_at: str
    gpu_index: int
    watchdog_triggered: bool
    results_path: str
    log_path: str
    total_samples_in_manifest: int
    total_result_lines_written: int


def _launch(spec: ModelSpec, manifest_path: str, results_path: str, log_path: str, gpu_index: int):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    env.setdefault("TMPDIR", os.environ.get("TMPDIR", "/tmp"))
    env.pop("HF_HOME", None) or env.update({k: v for k, v in os.environ.items() if k == "HF_HOME"})
    cmd = [
        spec.interpreter_path,
        spec.script_path,
        "--manifest",
        manifest_path,
        "--output",
        results_path,
        "--timeout-seconds",
        str(spec.timeout_seconds),
        "--max-new-tokens",
        str(spec.max_new_tokens),
    ]
    log_fh = open(log_path, "a")
    logger.info("[%s] launching on GPU %d: %s", spec.name, gpu_index, " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT)
    return proc, log_fh


def _write_run_meta(run_meta_path: str, result: RunResult) -> None:
    import json

    with open(run_meta_path, "w") as f:
        json.dump(result.__dict__, f, indent=2)


def run_model_subprocess(
    spec: ModelSpec, manifest_path: str, results_path: str, log_path: str, gpu_index: int,
    stall_minutes: int = 15,
) -> RunResult:
    """Blocking single-model run (used directly in sequential mode, and internally by
    run_models_scheduled for each slot in parallel mode)."""
    manifest = io_utils.read_jsonl(manifest_path)
    started_at = _utcnow_iso()
    proc, log_fh = _launch(spec, manifest_path, results_path, log_path, gpu_index)

    watchdog_triggered = False
    last_size = -1
    last_progress_time = time.monotonic()
    try:
        while True:
            ret = proc.poll()
            if ret is not None:
                break
            time.sleep(POLL_INTERVAL_SECONDS)
            cur_size = os.path.getsize(results_path) if os.path.exists(results_path) else 0
            if cur_size != last_size:
                last_size = cur_size
                last_progress_time = time.monotonic()
            elif (time.monotonic() - last_progress_time) > stall_minutes * 60:
                logger.warning(
                    "[%s] no progress in results file for %d minutes, killing (pid=%d)",
                    spec.name, stall_minutes, proc.pid,
                )
                proc.kill()
                proc.wait()
                watchdog_triggered = True
                break
    finally:
        log_fh.close()

    exit_code = proc.returncode
    ended_at = _utcnow_iso()
    written = len(io_utils.read_jsonl(results_path))
    result = RunResult(
        model_name=spec.name,
        exit_code=exit_code if exit_code is not None else -1,
        started_at=started_at,
        ended_at=ended_at,
        gpu_index=gpu_index,
        watchdog_triggered=watchdog_triggered,
        results_path=results_path,
        log_path=log_path,
        total_samples_in_manifest=len(manifest),
        total_result_lines_written=written,
    )
    run_meta_path = os.path.join(os.path.dirname(results_path), f"{spec.name}_run_meta.json")
    _write_run_meta(run_meta_path, result)
    logger.info("[%s] finished: %s", spec.name, result)
    return result


def run_models_scheduled(
    specs: list, manifest_path: str, output_root: str, parallel: bool = True,
    stall_minutes: int = 15, min_free_mb: int = 20000, max_util_pct: int = 30,
) -> list:
    """Runs every model in `specs` against the same manifest. If parallel=False, runs them one
    at a time (simplest, safest on a shared cluster). If parallel=True, launches as many as
    there are currently-idle GPUs at once (each GPU reserved at launch time to avoid two
    subprocesses racing onto the same card), and launches the rest as GPUs free up.
    """
    if not parallel:
        results = []
        for spec in specs:
            model_dir = os.path.join(output_root, spec.name)
            os.makedirs(model_dir, exist_ok=True)
            results_path = os.path.join(model_dir, "results.jsonl")
            log_path = os.path.join(model_dir, "run.log")
            gpu_index = select_idle_gpu(min_free_mb=min_free_mb, max_util_pct=max_util_pct)
            results.append(
                run_model_subprocess(spec, manifest_path, results_path, log_path, gpu_index, stall_minutes)
            )
        return results

    # Parallel scheduling: non-blocking Popen for every launched model, single poll loop.
    pending = list(specs)
    running = {}  # model_name -> dict(proc, log_fh, spec, gpu_index, results_path, log_path,
                  #                    started_at, last_size, last_progress_time)
    reserved_gpus = set()
    completed = []

    def try_launch_pending():
        still_pending = []
        for spec in pending:
            try:
                gpu_index = select_idle_gpu(
                    min_free_mb=min_free_mb, max_util_pct=max_util_pct, exclude=tuple(reserved_gpus)
                )
            except RuntimeError:
                still_pending.append(spec)
                continue
            model_dir = os.path.join(output_root, spec.name)
            os.makedirs(model_dir, exist_ok=True)
            results_path = os.path.join(model_dir, "results.jsonl")
            log_path = os.path.join(model_dir, "run.log")
            proc, log_fh = _launch(spec, manifest_path, results_path, log_path, gpu_index)
            reserved_gpus.add(gpu_index)
            running[spec.name] = dict(
                proc=proc, log_fh=log_fh, spec=spec, gpu_index=gpu_index,
                results_path=results_path, log_path=log_path, started_at=_utcnow_iso(),
                last_size=-1, last_progress_time=time.monotonic(),
            )
        pending[:] = still_pending

    manifest = io_utils.read_jsonl(manifest_path)
    try_launch_pending()
    while pending or running:
        time.sleep(POLL_INTERVAL_SECONDS)
        finished_names = []
        for name, state in running.items():
            proc = state["proc"]
            ret = proc.poll()
            results_path = state["results_path"]
            cur_size = os.path.getsize(results_path) if os.path.exists(results_path) else 0
            watchdog_triggered = False

            if ret is None:
                if cur_size != state["last_size"]:
                    state["last_size"] = cur_size
                    state["last_progress_time"] = time.monotonic()
                    continue  # progress was made; still running, not finished this cycle
                elif (time.monotonic() - state["last_progress_time"]) > stall_minutes * 60:
                    logger.warning(
                        "[%s] no progress for %d minutes, killing (pid=%d)",
                        name, stall_minutes, proc.pid,
                    )
                    proc.kill()
                    proc.wait()
                    watchdog_triggered = True
                    ret = proc.returncode
                else:
                    continue  # still running normally, not finished this cycle

            # Process has finished (naturally or via watchdog kill)
            state["log_fh"].close()
            written = len(io_utils.read_jsonl(results_path))
            result = RunResult(
                model_name=name,
                exit_code=ret if ret is not None else -1,
                started_at=state["started_at"],
                ended_at=_utcnow_iso(),
                gpu_index=state["gpu_index"],
                watchdog_triggered=watchdog_triggered,
                results_path=results_path,
                log_path=state["log_path"],
                total_samples_in_manifest=len(manifest),
                total_result_lines_written=written,
            )
            run_meta_path = os.path.join(os.path.dirname(results_path), f"{name}_run_meta.json")
            _write_run_meta(run_meta_path, result)
            logger.info("[%s] finished: %s", name, result)
            completed.append(result)
            reserved_gpus.discard(state["gpu_index"])
            finished_names.append(name)

        for name in finished_names:
            del running[name]

        if pending:
            try_launch_pending()

    return completed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("subprocess_runner module loaded OK")
