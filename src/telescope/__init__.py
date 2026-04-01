"""
Telescope: RL post-training for LLMs.

A modular framework for reinforcement learning based post-training
with asynchronous inference and training.
"""

__version__ = "0.1.2"
schema_version = "0.3.0"

# Per-table schema versions for parquet tables uploaded to wandb.
# Bump the version for a table whenever its schema (columns, types, semantics) changes.
table_schema_versions = {
    # events zip tables
    "orchestrator": "0.2",
    "trainer": "0.2",
    "events_rollout": "0.1",
    "events_infra": "0.1",
    "gpu": "0.1",
    "cpu": "0.1",
    "vllm": "0.1",
    "logs": "0.1",
    # steps zip tables
    "prompts": "0.1",
    "generations": "0.1",
    "env_responses": "0.1",
    "tool_calls": "0.1",
    "sandboxes": "0.1",
    "samples_data": "0.3",
    "rollouts_metrics": "0.2",
    "turn_metrics": "0.1",
    "golden_answers": "0.2",
    "info_turns": "0.2",
    "sample_tags": "0.2",
    "metrics": "0.2",
    # discarded tables (events zip)
    "prompts_discarded": "0.1",
    "generations_discarded": "0.1",
    "env_responses_discarded": "0.1",
    "tool_calls_discarded": "0.1",
    "samples_data_discarded": "0.3",
    "rollouts_metrics_discarded": "0.2",
    "golden_answers_discarded": "0.2",
    "info_turns_discarded": "0.2",
    "sample_tags_discarded": "0.2",
    # eval tables (events zip)
    "prompts_eval": "0.1",
    "generations_eval": "0.1",
    "env_responses_eval": "0.1",
    "tool_calls_eval": "0.1",
    "samples_data_eval": "0.2",
    "rollouts_metrics_eval": "0.2",
    "golden_answers_eval": "0.2",
    "info_turns_eval": "0.2",
    "sample_tags_eval": "0.2",
}
