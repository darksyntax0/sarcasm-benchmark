"""JSONL helpers used on both sides of the orchestrator/model-subprocess boundary."""
import json
import os


def append_jsonl(path: str, record: dict) -> None:
    """Append one JSON line, flushed and fsync'd immediately so partial progress survives a
    crash or a watchdog kill during a multi-hour unattended run on a shared cluster."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: str) -> list:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def read_processed_sample_ids(results_jsonl_path: str) -> set:
    """Powers resume: a killed/restarted model subprocess skips sample_ids already written."""
    ids = set()
    for record in read_jsonl(results_jsonl_path):
        sid = record.get("sample_id")
        if sid:
            ids.add(sid)
    return ids
