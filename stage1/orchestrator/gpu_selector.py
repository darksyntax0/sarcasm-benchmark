"""Live GPU status querying and idle-GPU selection. Never cache results across a run -- this
cluster is shared with other users whose jobs come and go between checks."""
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuStatus:
    index: int
    name: str
    mem_used_mb: float
    mem_total_mb: float
    util_pct: float

    @property
    def mem_free_mb(self) -> float:
        return self.mem_total_mb - self.mem_used_mb


def query_gpu_status() -> list:
    """Runs nvidia-smi fresh, parses CSV output. Raises RuntimeError if nvidia-smi is
    unavailable or returns unparseable output -- this must fail loudly, not silently proceed
    with stale/guessed GPU state."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise RuntimeError(f"nvidia-smi query failed: {e}")

    statuses = []
    for line in proc.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, mem_used, mem_total, util = parts
        statuses.append(
            GpuStatus(
                index=int(index),
                name=name,
                mem_used_mb=float(mem_used),
                mem_total_mb=float(mem_total),
                util_pct=float(util),
            )
        )
    if not statuses:
        raise RuntimeError(f"nvidia-smi returned no parseable GPU rows: {proc.stdout!r}")
    return statuses


def select_idle_gpu(min_free_mb: int = 20000, max_util_pct: int = 30, exclude=()) -> int:
    """Picks the GPU with the most free memory among those meeting the idle thresholds and not
    in `exclude` (used by the parallel scheduler to avoid double-booking a GPU it already
    reserved for another model this run). Raises RuntimeError if none qualify."""
    statuses = query_gpu_status()
    candidates = [
        g
        for g in statuses
        if g.mem_free_mb >= min_free_mb and g.util_pct <= max_util_pct and g.index not in exclude
    ]
    if not candidates:
        raise RuntimeError(
            f"no idle GPU meeting thresholds (min_free_mb={min_free_mb}, "
            f"max_util_pct={max_util_pct}, exclude={exclude}): {statuses}"
        )
    return max(candidates, key=lambda g: g.mem_free_mb).index


if __name__ == "__main__":
    for g in query_gpu_status():
        print(g)
    print("selected idle GPU:", select_idle_gpu())
