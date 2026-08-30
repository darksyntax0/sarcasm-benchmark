"""Stdlib-only data contracts shared between the orchestrator env and all 5 model conda envs.

No third-party imports here (not even yaml/pandas) so this module can be imported unmodified
by every environment involved in a benchmark run.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass(frozen=True)
class ManifestRecord:
    """One row of a manifest file handed to a model subprocess. Deliberately excludes ground
    truth -- the model side must never see labels."""

    sample_id: str
    target_wav_path: str
    target_text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ManifestRecord":
        return ManifestRecord(
            sample_id=d["sample_id"],
            target_wav_path=d["target_wav_path"],
            target_text=d["target_text"],
        )


@dataclass(frozen=True)
class GroundTruthRecord:
    """One row of ground_truth.jsonl. Orchestrator-only; never read by model subprocesses."""

    sample_id: str
    gt_category: str
    gt_sarcasm_source: str
    gt_sarcasm_target: str
    gt_explanation: str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "GroundTruthRecord":
        return GroundTruthRecord(
            sample_id=d["sample_id"],
            gt_category=d.get("gt_category", ""),
            gt_sarcasm_source=d.get("gt_sarcasm_source", ""),
            gt_sarcasm_target=d.get("gt_sarcasm_target", ""),
            gt_explanation=d.get("gt_explanation", ""),
        )


@dataclass
class ResultRecord:
    """One line of a model's results.jsonl. Fields default to None/empty so a partially-failed
    sample (timeout, parse error, audio error) still produces a well-formed record."""

    sample_id: str
    model_name: str
    predicted_category: Optional[str] = None
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    reasoning_steps: Optional[list] = None
    raw_output: str = ""
    parse_error: Optional[str] = None
    audio_input_error: Optional[str] = None
    error: Optional[str] = None
    latency_seconds: Optional[float] = None
    gpu_mem_allocated_mb: Optional[float] = None
    gpu_mem_reserved_mb: Optional[float] = None
    timestamp: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "ResultRecord":
        # Use .get() with defaults throughout -- real model output / partial writes are messy,
        # this must never raise on a record missing a key.
        return ResultRecord(
            sample_id=d.get("sample_id", ""),
            model_name=d.get("model_name", ""),
            predicted_category=d.get("predicted_category"),
            confidence=d.get("confidence"),
            explanation=d.get("explanation"),
            reasoning_steps=d.get("reasoning_steps"),
            raw_output=d.get("raw_output", ""),
            parse_error=d.get("parse_error"),
            audio_input_error=d.get("audio_input_error"),
            error=d.get("error"),
            latency_seconds=d.get("latency_seconds"),
            gpu_mem_allocated_mb=d.get("gpu_mem_allocated_mb"),
            gpu_mem_reserved_mb=d.get("gpu_mem_reserved_mb"),
            timestamp=d.get("timestamp", ""),
        )
