"""训练适配器：Agent Lightning rollout 以及 SFT/GRPO 支持。"""
from .grpo_config import GRPOConfig, load_grpo_config
from .grpo_dataset import EpisodeSource, GRPODatasetBundle, GRPODatasetError, load_grpo_dataset
from .lightning_agent import build_lightning_agent, run_dispute_rollout
from .monitor import CollapseMonitor

__all__ = [
    "CollapseMonitor",
    "EpisodeSource",
    "GRPOConfig",
    "GRPODatasetBundle",
    "GRPODatasetError",
    "build_lightning_agent",
    "load_grpo_dataset",
    "load_grpo_config",
    "run_dispute_rollout",
]
