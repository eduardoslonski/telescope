"""Convert native training checkpoints (DCP / Megatron) to HuggingFace format.

Extracted from ``tools/convert_checkpoint_to_hf.py`` so it can be imported by
the standalone eval driver for on-the-fly conversion.
"""
from __future__ import annotations

import json
import pickle as _pickle
import re
import shutil
from pathlib import Path

import torch
from torch.distributed.checkpoint import FileSystemReader


# ---------------------------------------------------------------------------
# Megatron pickle compatibility
# ---------------------------------------------------------------------------


class _SafeUnpickler(_pickle.Unpickler):
    """Unpickler that stubs out Megatron-specific classes.

    DCP metadata files may reference Megatron classes (e.g. ``TransformerConfig``)
    that aren't available at conversion time.  Replace them with dummy objects so
    the metadata can still be deserialized.
    """

    def find_class(self, mod_name: str, name: str):
        if mod_name.startswith("megatron"):

            class _Dummy:
                def __init__(self, *a, **kw):
                    pass

            return _Dummy
        return super().find_class(mod_name, name)


class _MegatronStorageReader(FileSystemReader):
    """FileSystemReader that handles Megatron-specific pickle classes."""

    def read_metadata(self):
        path = self.fs.concat_path(self.path, ".metadata")
        with self.fs.create_stream(path, "rb") as f:
            metadata = _SafeUnpickler(f).load()
        if getattr(metadata, "storage_meta", None) is None:
            from torch.distributed.checkpoint import StorageMeta
            metadata.storage_meta = StorageMeta()
        metadata.storage_meta.load_id = self.load_id
        if metadata.planner_data is None:
            metadata.planner_data = {}
        return metadata


# ---------------------------------------------------------------------------
# FSDP (DCP) conversion
# ---------------------------------------------------------------------------

def _convert_fsdp(ckpt_dir: Path, output_dir: Path, meta: dict) -> None:
    """Convert an FSDP DCP checkpoint to HuggingFace safetensors format.

    FSDP state dict keys are already in HF naming convention (the model is
    ``AutoModelForCausalLM``), so no name conversion is needed.  We just load
    the DCP shards into a flat state dict and save as safetensors.
    """
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    print(f"  Loading FSDP DCP checkpoint from {ckpt_dir} ...")

    # DCP stores model weights under the "model" key.
    # We use the low-level _load_state_dict with no_dist=True so we don't
    # need torch.distributed to be initialized.
    reader = FileSystemReader(str(ckpt_dir))
    metadata = reader.read_metadata()

    # Discover all tensor keys that belong to the model (skip optimizer, step)
    model_keys: list[str] = []
    for key in metadata.state_dict_metadata:
        if key.startswith("model."):
            model_keys.append(key)

    if not model_keys:
        raise RuntimeError(
            f"No 'model.*' keys found in DCP metadata at {ckpt_dir}. "
            f"Available keys: {list(metadata.state_dict_metadata.keys())[:20]}"
        )

    # Build skeleton state dict with empty tensors matching the stored shapes/dtypes
    state_dict: STATE_DICT_TYPE = {}
    for key in model_keys:
        tensor_meta = metadata.state_dict_metadata[key]
        # TensorStorageMetadata has .size and .properties.dtype
        state_dict[key] = torch.empty(
            tensor_meta.size,
            dtype=tensor_meta.properties.dtype,
        )

    _load_state_dict(
        state_dict,
        storage_reader=reader,
        no_dist=True,
    )

    # Strip the "model." prefix that DCP used for nesting
    hf_state_dict: dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        hf_key = key.removeprefix("model.")
        hf_state_dict[hf_key] = tensor

    _save_hf_checkpoint(hf_state_dict, ckpt_dir, output_dir)
    print(f"  FSDP checkpoint converted: {output_dir}")


# ---------------------------------------------------------------------------
# Megatron TP layout helpers
# ---------------------------------------------------------------------------

def _detect_tp_size(metadata, meta: dict) -> int:
    """Detect tensor parallelism size for a Megatron DCP checkpoint.

    Reads ``tp_size`` from *meta* (written by newer checkpoints) and falls
    back to counting DCP shard chunks on a column-parallel tensor.
    """
    # Preferred: explicit value stored at save time.
    tp = meta.get("tp_size")
    if tp is not None and int(tp) >= 1:
        return int(tp)

    # Fallback: inspect DCP metadata.  Column-parallel tensors have one chunk
    # per TP rank.  Embedding weight is ideal because it exists on exactly one
    # PP stage (no PP duplication) and is always column-parallel.
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    for key, tmeta in metadata.state_dict_metadata.items():
        if not isinstance(tmeta, TensorStorageMetadata):
            continue
        if "word_embeddings.weight" in key or "output_layer.weight" in key:
            if hasattr(tmeta, "chunks") and tmeta.chunks:
                return len(tmeta.chunks)

    # Second pass: try any linear_qkv weight (also column-parallel).
    for key, tmeta in metadata.state_dict_metadata.items():
        if not isinstance(tmeta, TensorStorageMetadata):
            continue
        if "linear_qkv.weight" in key:
            if hasattr(tmeta, "chunks") and tmeta.chunks:
                return len(tmeta.chunks)

    return 1


def _deinterleave_tp_gated_mlp(param: torch.Tensor, tp_size: int) -> torch.Tensor:
    """Fix the interleaved gate+up layout produced by DCP reconstruction of TP>1.

    With TP>1, each rank stores ``[gate_shard_i; up_shard_i]`` (concatenated
    along dim 0).  DCP ``no_dist=True`` places shards at their recorded offsets,
    producing::

        [gate_0; up_0; gate_1; up_1; ...]

    ``_convert_qwen_to_hf`` expects the canonical layout::

        [gate_full; up_full]

    This function re-arranges the rows so that ``chunk(2, dim=0)`` yields the
    correct gate and up projections.
    """
    if tp_size <= 1:
        return param
    shard_size = param.shape[0] // tp_size
    shards = param.split(shard_size, dim=0)
    gates, ups = [], []
    for shard in shards:
        g, u = shard.chunk(2, dim=0)
        gates.append(g)
        ups.append(u)
    return torch.cat(gates + ups, dim=0)


# ---------------------------------------------------------------------------
# Megatron conversion
# ---------------------------------------------------------------------------

def _convert_megatron(ckpt_dir: Path, output_dir: Path, meta: dict) -> None:
    """Convert a Megatron dist_checkpointing checkpoint to HF safetensors.

    Loads the Megatron dist_checkpointing state dict without distributed
    setup (``no_dist=True``), converts Megatron parameter names to HF
    format via ``_convert_qwen_to_hf``, and saves as safetensors.

    Handles TP>1 checkpoints by detecting the tensor parallelism degree and
    deinterleaving gated-MLP weights (``linear_fc1``) whose gate and up
    projections are interleaved across TP shards in the DCP layout.

    PP>1 and EP>1 require the distributed runtime for conversion
    (``gather_weights_for_inference``); the offline path does not support them.
    """
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    print(f"  Loading Megatron checkpoint from {ckpt_dir} ...")

    # Warn about PP / EP limitations early.
    pp_size = meta.get("pp_size", 1)
    ep_size = meta.get("ep_size", 1)
    if int(pp_size) > 1:
        print(
            f"  WARNING: Checkpoint was saved with PP={pp_size}. "
            "Offline conversion may produce incorrect results. "
            "Use the distributed runtime (gather_weights_for_inference) instead."
        )
    if int(ep_size) > 1:
        print(
            f"  WARNING: Checkpoint was saved with EP={ep_size}. "
            "Offline conversion may produce incorrect results for expert weights. "
            "Use the distributed runtime (gather_weights_for_inference) instead."
        )

    reader = _MegatronStorageReader(str(ckpt_dir))
    metadata = reader.read_metadata()

    tp_size = _detect_tp_size(metadata, meta)
    if tp_size > 1:
        print(f"  Detected TP={tp_size}; will deinterleave gated-MLP weights.")

    # Collect model keys (skip optimizer, step, etc.)
    model_keys: list[str] = []
    for key in metadata.state_dict_metadata:
        if key.startswith("model.") and not key.startswith("model.optimizer"):
            model_keys.append(key)

    if not model_keys:
        raise RuntimeError(
            f"No model keys found in Megatron checkpoint at {ckpt_dir}. "
            f"Available keys: {list(metadata.state_dict_metadata.keys())[:20]}"
        )

    # Build skeleton state dict
    state_dict: STATE_DICT_TYPE = {}
    for key in model_keys:
        tensor_meta = metadata.state_dict_metadata[key]
        state_dict[key] = torch.empty(
            tensor_meta.size,
            dtype=tensor_meta.properties.dtype,
        )

    _load_state_dict(
        state_dict,
        storage_reader=reader,
        no_dist=True,
    )

    # Load HF config to get model architecture info for name conversion
    from transformers import AutoConfig

    hf_meta_dir = ckpt_dir / "hf_meta"
    hf_config = AutoConfig.from_pretrained(str(hf_meta_dir), trust_remote_code=True)

    vocab_size = meta.get("vocab_size", hf_config.vocab_size)

    # Convert Megatron names to HF names
    # Import the conversion function from the megatron backend
    from telescope.trainer.backends.megatron import _convert_qwen_to_hf

    # Regex for any linear_fc1 weight (dense MLP, MoE expert, shared expert).
    _gated_mlp_re = re.compile(r"linear_fc1\.weight$")

    hf_state_dict: dict[str, torch.Tensor] = {}
    for key, param in state_dict.items():
        # Strip the DCP nesting prefix (e.g., "model." from dist_checkpointing)
        # and add the "module.module." prefix that _convert_qwen_to_hf expects
        # (matching the DDP(Float16Module(GPTModel)) naming from named_parameters()).
        name = key
        if name.startswith("model."):
            name = name[len("model."):]
        name = f"module.module.{name}"

        # Remove vocab padding
        if "output_layer.weight" in name or "word_embeddings.weight" in name:
            param = param[:vocab_size, :]

        # Fix gated-MLP interleaving from TP>1 DCP reconstruction.
        # DCP places TP shards contiguously: [gate_0;up_0; gate_1;up_1; ...]
        # but _convert_qwen_to_hf expects [gate_full; up_full].
        if tp_size > 1 and _gated_mlp_re.search(name):
            param = _deinterleave_tp_gated_mlp(param, tp_size)

        try:
            hf_params = _convert_qwen_to_hf(
                name=name,
                param=param,
                num_attention_heads=hf_config.num_attention_heads,
                num_key_value_heads=getattr(
                    hf_config, "num_key_value_heads",
                    hf_config.num_attention_heads,
                ),
                hidden_size=hf_config.hidden_size,
                head_dim=getattr(hf_config, "head_dim", None),
            )
            for hf_name, hf_param in hf_params:
                hf_state_dict[hf_name] = hf_param
        except ValueError:
            print(f"  WARNING: Skipping unknown Megatron param: {name}")

    _save_hf_checkpoint(hf_state_dict, ckpt_dir, output_dir)
    print(f"  Megatron checkpoint converted: {output_dir}")


# ---------------------------------------------------------------------------
# Shared save logic
# ---------------------------------------------------------------------------

def _save_hf_checkpoint(
    hf_state_dict: dict[str, torch.Tensor],
    ckpt_dir: Path,
    output_dir: Path,
) -> None:
    """Save HF state dict as safetensors and copy hf_meta files."""
    from safetensors.torch import save_file

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check total size to decide if we need sharding
    total_bytes = sum(t.numel() * t.element_size() for t in hf_state_dict.values())
    shard_size = 5 * 1024**3  # 5GB per shard

    if total_bytes <= shard_size:
        # Single file
        save_file(hf_state_dict, output_dir / "model.safetensors")
    else:
        # Shard using huggingface_hub
        from huggingface_hub import split_torch_state_dict_into_shards

        plan = split_torch_state_dict_into_shards(
            hf_state_dict, max_shard_size="5GB"
        )
        for filename, shard_tensors in plan.filename_to_tensors.items():
            shard = {k: hf_state_dict[k] for k in shard_tensors}
            save_file(shard, output_dir / filename)

        # Write index
        index = {
            "metadata": {"total_size": total_bytes},
            "weight_map": plan.tensor_to_filename,
        }
        (output_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2)
        )

    # Copy hf_meta files (config.json, tokenizer, etc.)
    hf_meta_dir = ckpt_dir / "hf_meta"
    if hf_meta_dir.is_dir():
        for f in hf_meta_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, output_dir / f.name)
    else:
        print(f"  WARNING: No hf_meta/ found in {ckpt_dir}. "
              "Output will be missing config.json and tokenizer files.")

    param_count = sum(t.numel() for t in hf_state_dict.values())
    print(f"  Saved {len(hf_state_dict)} tensors, "
          f"{param_count/1e6:.1f}M params, "
          f"{total_bytes/1024**3:.2f} GB")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def convert_single(ckpt_dir: Path, output_dir: Path) -> None:
    """Convert a single checkpoint directory."""
    meta_path = ckpt_dir / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(
            f"No meta.json found in {ckpt_dir}. "
            "Is this a telescope training checkpoint?"
        )

    meta = json.loads(meta_path.read_text())
    backend = meta.get("backend", "unknown")

    print(f"Converting step {meta.get('step', '?')} ({backend} backend)")

    if backend == "fsdp":
        _convert_fsdp(ckpt_dir, output_dir, meta)
    elif backend == "megatron":
        _convert_megatron(ckpt_dir, output_dir, meta)
    else:
        raise ValueError(f"Unknown backend in meta.json: {backend!r}")
