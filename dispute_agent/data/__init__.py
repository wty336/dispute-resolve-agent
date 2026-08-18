"""合成数据生成、渲染、切分与校验。"""
from .generator import FactInstance, generate_fact_instances
from .renderer import render_observation, render_sft_trace
from .splits import DatasetManifest, HumanAudit, SplitManifest, build_dataset_manifest
from .validators import validate_observation, validate_trace_messages

__all__ = [
    "DatasetManifest",
    "FactInstance",
    "HumanAudit",
    "SplitManifest",
    "build_dataset_manifest",
    "generate_fact_instances",
    "render_observation",
    "render_sft_trace",
    "validate_observation",
    "validate_trace_messages",
]
