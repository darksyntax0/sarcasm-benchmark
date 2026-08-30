"""Loads and validates configs/model_registry.yaml. Runs only in the benchmark-orchestrator env."""
import os
from dataclasses import dataclass

import yaml

_FRAMEWORK_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    interpreter_path: str
    script_path: str
    timeout_seconds: int
    max_new_tokens: int


def load_model_registry(path: str = None) -> dict:
    """Parses the YAML registry, returns {name: ModelSpec}. Validates -- fail-fast, before any
    GPU work -- that every interpreter_path and script_path actually exists on disk."""
    if path is None:
        path = os.path.join(_FRAMEWORK_ROOT, "configs", "model_registry.yaml")

    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    specs = {}
    errors = []
    for name, cfg in raw["models"].items():
        interpreter_path = os.path.expandvars(cfg["interpreter_path"])
        script_path = os.path.join(_FRAMEWORK_ROOT, cfg["script_path"])

        if not os.path.isfile(interpreter_path):
            errors.append(f"[{name}] interpreter_path does not exist: {interpreter_path}")
        if not os.path.isfile(script_path):
            errors.append(f"[{name}] script_path does not exist: {script_path}")

        specs[name] = ModelSpec(
            name=name,
            interpreter_path=interpreter_path,
            script_path=script_path,
            timeout_seconds=int(cfg["timeout_seconds"]),
            max_new_tokens=int(cfg["max_new_tokens"]),
        )

    if errors:
        raise FileNotFoundError(
            "model_registry.yaml validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return specs


if __name__ == "__main__":
    specs = load_model_registry()
    for name, spec in specs.items():
        print(name, "->", spec)
