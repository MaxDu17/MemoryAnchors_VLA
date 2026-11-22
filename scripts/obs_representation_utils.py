"""Shared helpers for extracting observation representations from openpi checkpoints."""

from __future__ import annotations

import dataclasses
import enum
import logging
import pathlib
import types

import einops
import jax
import jax.numpy as jnp
import numpy as np
import tqdm

import flax.nnx as nnx

import openpi.models.model as _model
import openpi.shared.download as download
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
from torch.utils.data import Subset

import openpi.training.data_loader as _data_loader
from openpi.models.pi0 import make_attn_mask
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.training.data_loader import DataLoaderImpl


class ReprMode(enum.Enum):
    """How to extract per-frame representation vectors."""

    PREFIX_EMBED = "prefix_embed"
    IMAGE_ONLY = "image_only"
    PREFIX_OUT = "prefix_out"


def init_logging() -> None:
    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s [%(levelname)s] %(message)s")


def infer_task_idx_from_checkpoint(checkpoint_dir: pathlib.Path) -> int | None:
    task_idx_files = sorted(checkpoint_dir.glob("task_idx_*.txt"))
    if not task_idx_files:
        return None
    return max(int(path.read_text().strip()) for path in task_idx_files)


def decode_prompts(obs: _model.Observation) -> list[str]:
    if obs.tokenized_prompt is None or obs.tokenized_prompt_mask is None:
        return ["<no tokenized_prompt>"] * int(obs.state.shape[0])

    tokenizer = PaligemmaTokenizer()
    prompts = np.asarray(jax.device_get(obs.tokenized_prompt))
    masks = np.asarray(jax.device_get(obs.tokenized_prompt_mask))
    decoded = []
    for i in range(prompts.shape[0]):
        valid = prompts[i][masks[i]].tolist()
        decoded.append(tokenizer._tokenizer.decode(valid))  # noqa: SLF001
    return decoded


def load_jax_model(config: _config.TrainConfig, checkpoint_path: pathlib.Path) -> _model.BaseModel:
    weight_path = checkpoint_path / "model.safetensors"
    if weight_path.exists():
        raise ValueError(
            f"PyTorch checkpoint detected at {weight_path}. "
            "This script currently supports JAX checkpoints only (params/ directory)."
        )
    params_path = checkpoint_path / "params"
    if not params_path.exists():
        raise FileNotFoundError(f"No params/ directory under {checkpoint_path}")
    model = config.model.load(_model.restore_params(params_path, dtype=jnp.bfloat16))
    model.eval()
    return model


def _masked_mean_pool(tokens: jax.Array, mask: jax.Array) -> jax.Array:
    weights = mask.astype(jnp.float32)[..., None]
    denom = jnp.maximum(jnp.sum(weights, axis=1), 1.0)
    return jnp.sum(tokens * weights, axis=1) / denom


def _extract_prefix_embed(self: _model.BaseModel, observation: _model.Observation) -> jax.Array:
    obs = _model.preprocess_observation(None, observation, train=False)
    tokens, mask, _ = self.embed_prefix(obs)
    return _masked_mean_pool(tokens, mask)


def _extract_image_only(self: _model.BaseModel, observation: _model.Observation) -> jax.Array:
    obs = _model.preprocess_observation(None, observation, train=False)
    tokens = []
    mask = []
    for name in obs.images:
        image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)
        tokens.append(image_tokens)
        mask.append(
            einops.repeat(
                obs.image_masks[name],
                "b -> b s",
                s=image_tokens.shape[1],
            )
        )
    return _masked_mean_pool(jnp.concatenate(tokens, axis=1), jnp.concatenate(mask, axis=1))


def _extract_prefix_out(self: _model.BaseModel, observation: _model.Observation) -> jax.Array:
    obs = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(obs)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    (prefix_out, _), _ = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    if prefix_out is None:
        raise RuntimeError("Expected prefix_out from prefix-only forward pass.")
    return _masked_mean_pool(prefix_out, prefix_mask)


_EXTRACT_FNS = {
    ReprMode.PREFIX_EMBED: _extract_prefix_embed,
    ReprMode.IMAGE_ONLY: _extract_image_only,
    ReprMode.PREFIX_OUT: _extract_prefix_out,
}


def make_extract_fn(model: _model.BaseModel, repr_mode: ReprMode):
    fn = _EXTRACT_FNS[repr_mode]
    bound = types.MethodType(fn, model)
    return nnx_utils.module_jit(bound)


def model_from_train_state(train_state) -> _model.BaseModel:
    model = nnx.merge(train_state.model_def, train_state.params)
    model.eval()
    return model


def resolve_norm_stats_root(
    config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | None = None,
) -> pathlib.Path:
    if checkpoint_dir is not None:
        ckpt_assets = checkpoint_dir / "assets"
        if ckpt_assets.exists() and any(ckpt_assets.iterdir()):
            return ckpt_assets
    return config.assets_dirs


def make_data_config(
    config: _config.TrainConfig,
    norm_stats_root: pathlib.Path,
    task_id: int,
) -> _config.DataConfig:
    if isinstance(config.data, _config.LeRobotLiberoSeqDataConfig):
        data_config = config.data.create(config.assets_dirs, config.model, task_id=task_id)
    else:
        data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.asset_id is None:
        raise ValueError("asset_id is required to load norm stats from checkpoint.")
    norm_stats = _checkpoints.load_norm_stats(norm_stats_root, data_config.asset_id)
    return dataclasses.replace(data_config, norm_stats=norm_stats)


def create_loader(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_samples: int | None = None,
) -> tuple[DataLoaderImpl, int]:
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = _data_loader.transform_dataset(dataset, data_config)
    n_samples = len(dataset)
    n_to_iterate = n_samples if max_samples is None else min(n_samples, max_samples)
    num_batches = (n_to_iterate + batch_size - 1) // batch_size
    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        shuffle=False,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework="jax",
        drop_last=False,
    )
    return DataLoaderImpl(data_config, loader), n_samples


def task_name_from_repo_id(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def resolve_checkpoint_output_dir(
    checkpoint_path: pathlib.Path,
    *,
    output_dir: str | None,
    output_base: str,
    suffix: str = "",
) -> pathlib.Path:
    if output_dir is not None:
        return pathlib.Path(output_dir).resolve()
    exp_name = checkpoint_path.parent.name
    step = checkpoint_path.name
    base = pathlib.Path(output_base) / exp_name / step
    if suffix:
        base = base / suffix
    return base.resolve()


def n_sequential_tasks(config: _config.TrainConfig) -> int:
    data_factory = config.data
    if hasattr(data_factory, "repo_ids") and data_factory.repo_ids:
        return len(data_factory.repo_ids)
    return 1


def extract_task_embeddings(
    model: _model.BaseModel,
    extract_fn,
    config: _config.TrainConfig,
    norm_stats_root: pathlib.Path,
    task_id: int,
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_samples: int | None = None,
) -> tuple[np.ndarray, _config.DataConfig, int]:
    """Extract per-frame embeddings for one task in sequential dataset order."""
    data_config = make_data_config(config, norm_stats_root, task_id)
    loader, n_samples = create_loader(
        config,
        data_config,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed,
        max_samples=max_samples,
    )
    n_to_extract = n_samples if max_samples is None else min(n_samples, max_samples)
    task_name = task_name_from_repo_id(data_config.repo_id or f"task_{task_id}")
    logging.info("Task %d (%s): extracting %d / %d frames", task_id, task_name, n_to_extract, n_samples)

    chunks: list[np.ndarray] = []
    frame_idx = 0
    for observation, _ in tqdm.tqdm(
        loader,
        desc=task_name,
        total=(n_to_extract + batch_size - 1) // batch_size,
    ):
        if frame_idx >= n_to_extract:
            break

        embeddings = np.asarray(jax.device_get(extract_fn(observation))).astype(np.float32)
        frames_remaining = n_to_extract - frame_idx
        if embeddings.shape[0] > frames_remaining:
            embeddings = embeddings[:frames_remaining]

        chunks.append(embeddings)
        frame_idx += embeddings.shape[0]

    return np.concatenate(chunks, axis=0), data_config, n_samples


def create_task_dataset(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
) -> _data_loader.Dataset:
    dataset = _data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    return _data_loader.transform_dataset(dataset, data_config)


def create_subset_loader(
    config: _config.TrainConfig,
    data_config: _config.DataConfig,
    indices: np.ndarray | list[int],
    *,
    batch_size: int,
    num_workers: int,
    seed: int,
    max_samples: int | None = None,
) -> tuple[DataLoaderImpl, int]:
    dataset = create_task_dataset(config, data_config)
    index_list = [int(i) for i in indices]
    if max_samples is not None:
        index_list = index_list[:max_samples]
    subset = Subset(dataset, index_list)
    n_samples = len(subset)
    num_batches = (n_samples + batch_size - 1) // batch_size
    loader = _data_loader.TorchDataLoader(
        subset,
        local_batch_size=batch_size,
        shuffle=False,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework="jax",
        drop_last=False,
    )
    return DataLoaderImpl(data_config, loader), n_samples


def resolve_checkpoint_path(checkpoint_dir: str) -> pathlib.Path:
    return pathlib.Path(download.maybe_download(checkpoint_dir)).resolve()
