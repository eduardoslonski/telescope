"""
Telescope: RL post-training for LLMs.

A modular framework for reinforcement learning based post-training
with asynchronous inference and training.
"""

__version__ = "0.1.2"
schema_version = "0.1.9"

# Per-table schema versions for parquet tables uploaded to wandb.
# Bump the version for a table whenever its schema (columns, types, semantics) changes.
table_schema_versions = {
    # events zip tables
    "orchestrator": "0.1",
    "trainer": "0.2",
    "inference": "0.5",
    "gpu": "0.1",
    "cpu": "0.1",
    "vllm": "0.1",
    "logs": "0.1",
    # steps zip tables
    "prompts": "0.1",
    "rollouts": "0.1",
    "samples_data": "0.2",
    "rollouts_metrics": "0.1",
    "golden_answers": "0.1",
    "info_turns": "0.1",
    "sample_tags": "0.1",
    "metrics": "0.2",
    # discarded tables (events zip)
    "prompts_discarded": "0.1",
    "rollouts_discarded": "0.1",
    "samples_data_discarded": "0.2",
    "rollouts_metrics_discarded": "0.1",
    "golden_answers_discarded": "0.1",
    "info_turns_discarded": "0.1",
    "sample_tags_discarded": "0.1",
    # eval tables (events zip)
    "prompts_eval": "0.1",
    "rollouts_eval": "0.1",
    "samples_data_eval": "0.1",
    "rollouts_metrics_eval": "0.1",
    "golden_answers_eval": "0.1",
    "info_turns_eval": "0.1",
    "sample_tags_eval": "0.1",
}

