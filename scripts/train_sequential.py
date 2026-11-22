"""Sequential / continual training with optional Experience Replay (ER).

Usage:
  uv run scripts/train_sequential.py pi0_libero_low_mem_finetune-libero_goal_sequential \\
    --exp-name=my_run --overwrite --cl er

Dry run (few steps per task, no wandb, no checkpoints):
  uv run scripts/train_sequential.py pi0_libero_low_mem_finetune-libero_goal_sequential \\
    --exp-name=debug_run --overwrite --cl er --debug
"""

import os

# Quiet TensorFlow/XLA stderr noise (especially from DataLoader worker subprocesses).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import pathlib
import sys

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import dataclasses
import functools
import json
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tqdm_loggable.auto as tqdm
import wandb

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.libero_task_order as _libero_task_order
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
from openpi.training.data_loader import (
    MixedDataLoader,
    build_replay_subsets,
    create_deterministic_buffer,
    resample_replay_buffer_lst,
    create_full_task_datasets,
    create_torch_data_loader_replay_buffer,
)
from anchor_memory_pipeline import anchor_config_from_train_config, run_anchor_memory_pipeline
from obs_representation_utils import model_from_train_state, resolve_norm_stats_root

# Monkey-patch to fix 'List' feature type error in older LeRobot datasets.
try:
    import datasets.features.features as features

    _OLD_GENERATE_FROM_DICT = features.generate_from_dict

    def _new_generate_from_dict(obj):
        if isinstance(obj, dict) and obj.get("_type") == "List":
            obj["_type"] = "Sequence"
        return _OLD_GENERATE_FROM_DICT(obj)

    features.generate_from_dict = _new_generate_from_dict
except (ImportError, AttributeError):
    pass


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def _task_repo_id(config: _config.TrainConfig, canonical_task_id: int) -> str:
    """LeRobot repo id for canonical LIBERO task index."""
    data = config.data
    if hasattr(data, "repo_ids") and data.repo_ids:
        assert 0 <= canonical_task_id < len(data.repo_ids), (
            f"canonical_task_id {canonical_task_id} out of range for "
            f"{len(data.repo_ids)} repo_ids"
        )
        return data.repo_ids[canonical_task_id]
    return config.data.create(config.assets_dirs, config.model, task_id=canonical_task_id).repo_id


TASK_ORDER_FILENAME = "task_order.json"


def _resolve_training_task_order(config: _config.TrainConfig) -> list[int]:
    order = _libero_task_order.resolve_task_order(
        config.n_tasks,
        task_order_index=config.task_order_index,
        task_order_permutation=config.task_order_permutation,
    )
    assert len(order) == config.n_tasks
    return order


def _persist_task_order(
    checkpoint_dir: pathlib.Path,
    *,
    task_order_index: int | None,
    order: list[int],
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "task_order_index": task_order_index,
        "order": order,
    }
    (checkpoint_dir / TASK_ORDER_FILENAME).write_text(json.dumps(payload, indent=2) + "\n")


def _load_task_order_or_none(checkpoint_dir: pathlib.Path) -> list[int] | None:
    path = checkpoint_dir / TASK_ORDER_FILENAME
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return [int(x) for x in payload["order"]]


def _assert_task_order_matches_checkpoint(
    checkpoint_dir: pathlib.Path,
    expected_order: list[int],
) -> None:
    saved_order = _load_task_order_or_none(checkpoint_dir)
    if saved_order is None:
        return
    assert saved_order == expected_order, (
        f"Checkpoint task order {saved_order} does not match requested order {expected_order}. "
        f"See {checkpoint_dir / TASK_ORDER_FILENAME}"
    )


def _assert_loader_matches_canonical_task(
    config: _config.TrainConfig,
    loader: _data_loader.DataLoader,
    *,
    stage: int,
    canonical_task_id: int,
) -> None:
    expected_repo = _task_repo_id(config, canonical_task_id)
    actual_repo = loader.data_config().repo_id
    assert actual_repo == expected_repo, (
        f"Stage {stage}: loader repo {actual_repo!r} != expected canonical task "
        f"{canonical_task_id} repo {expected_repo!r}"
    )


def _batch_size_from_obs(obs: _model.Observation) -> int:
    """Batch size inferred from observation tensors (used for ER debug logging)."""
    if obs.state is not None:
        return int(obs.state.shape[0])
    return int(next(iter(obs.images.values())).shape[0])


def _decode_sample_prompts(obs: _model.Observation, n: int = 3, *, start: int = 0) -> list[str]:
    """Decode up to `n` tokenized prompts from batch slice [start, start+n) for debug logs."""
    if obs.tokenized_prompt is None or obs.tokenized_prompt_mask is None:
        return ["<no tokenized_prompt>"]

    from openpi.models.tokenizer import PaligemmaTokenizer

    tokenizer = PaligemmaTokenizer()
    prompts = np.asarray(jax.device_get(obs.tokenized_prompt))
    masks = np.asarray(jax.device_get(obs.tokenized_prompt_mask))
    decoded = []
    for i in range(start, min(start + n, prompts.shape[0])):
        valid = prompts[i][masks[i]].tolist()
        decoded.append(tokenizer._tokenizer.decode(valid))  # noqa: SLF001
    return decoded


def _prompt_signatures(obs: _model.Observation, start: int, end: int) -> set[tuple[int, ...]]:
    """Set of token-id tuples for prompts in [start, end); used to detect current/replay overlap."""
    if obs.tokenized_prompt is None or obs.tokenized_prompt_mask is None:
        return set()
    prompts = np.asarray(jax.device_get(obs.tokenized_prompt))
    masks = np.asarray(jax.device_get(obs.tokenized_prompt_mask))
    return {tuple(prompts[i][masks[i]].tolist()) for i in range(start, end)}


def _log_er_replay_buffer(
    config: _config.TrainConfig,
    *,
    buffer_lst: list[list[int]],
    task_order: list[int],
) -> None:
    """Log per-task replay index counts. `buffer_lst[i]` = local dataset indices for task `task_order[i]`."""
    logging.info("ER replay buffer (per-task memory indices):")
    for buf_idx, canonical_task_id in enumerate(task_order[: len(buffer_lst)]):
        repo = _task_repo_id(config, canonical_task_id)
        short_repo = repo.rsplit("/", 1)[-1]
        logging.info(
            "  stage %d (canonical task %d): %d/%d samples -> %s",
            buf_idx,
            canonical_task_id,
            len(buffer_lst[buf_idx]),
            config.n_memories,
            short_repo,
        )


def _log_er_replay_loader(
    config: _config.TrainConfig,
    *,
    dataset_lst: list[Any],
    replay_task_indices: list[int],
) -> None:
    """Log replay ConcatDataset size. `dataset_lst[i]` corresponds to `replay_task_indices[i]`."""
    total = sum(len(dataset) for dataset in dataset_lst)
    task_summaries = [
        f"t{task_idx}({len(dataset_lst[i])})"
        for i, task_idx in enumerate(replay_task_indices[: len(dataset_lst)])
    ]
    logging.info(
        "ER replay loader: batch_size=%d, tasks=[%s], total_replay_samples=%d",
        config.batch_size,
        ", ".join(task_summaries),
        total,
    )


def _uses_er(config: _config.TrainConfig) -> bool:
    """True when continual learning uses an experience-replay buffer (random or anchor-selected)."""
    return config.cl in ("er", "er_anchor")


_VALID_CL_MODES = frozenset({"", "er", "er_anchor"})


def _validate_cl_mode(config: _config.TrainConfig) -> None:
    if config.cl not in _VALID_CL_MODES:
        raise ValueError(
            f"Invalid --cl {config.cl!r}; must be one of: '', 'er', 'er_anchor'"
        )


def _replay_task_slice_end(task_idx: int) -> int:
    """Exclusive end index into replay slice when training task `task_idx` (past tasks only; empty at task 0)."""
    return task_idx


def _replay_task_ids(order: list[int], replay_end: int) -> list[int]:
    """Canonical LIBERO task id for each slot in the replay slice (length `replay_end`)."""
    assert replay_end <= len(order)
    return list(order[:replay_end])


def _canonical_task_id(order: list[int], stage: int) -> int:
    assert 0 <= stage < len(order)
    return order[stage]


def _build_replay_loader(
    config: _config.TrainConfig,
    dataset_lst: list[Any],
    buffer_lst: list[list[int]],
    *,
    replay_end: int,
    data_config,
    data_sharding,
):
    """Build a replay DataLoader over past-task subsets, or None when there is no replay yet."""
    if replay_end == 0:
        return None
    subsets = build_replay_subsets(dataset_lst[:replay_end], buffer_lst[:replay_end])
    return create_torch_data_loader_replay_buffer(
        subsets,
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=data_sharding,
        shuffle=True,
        num_workers=config.num_workers,
        seed=config.seed,
    )


def _reference_task_ids_for_anchor(config: _config.TrainConfig, stage: int, order: list[int]) -> list[int]:
    """Canonical LIBERO task ids seen before training stage `stage` (i.e. ``order[:stage]``)."""
    del config
    return list(order[:stage])


def _run_anchor_selection_for_task(
    config: _config.TrainConfig,
    train_state,
    *,
    stage: int,
    order: list[int],
    buffer_lst: list[list[int]],
) -> None:
    """Run anchor-memory pipeline before stage `stage` and write selected indices into `buffer_lst`.

    Uses the current `train_state` model, saves artifacts under `checkpoint_dir/anchor_memories/`,
    then writes anchor selections for past tasks into `buffer_lst[:replay_end]`.
    """
    reference_task_ids = _reference_task_ids_for_anchor(config, stage, order)
    if not reference_task_ids:
        return

    replay_end = _replay_task_slice_end(stage)
    next_canonical_task_id = _canonical_task_id(order, stage)

    model = model_from_train_state(train_state)
    norm_stats_root = resolve_norm_stats_root(config, config.checkpoint_dir)
    anchor_cfg = anchor_config_from_train_config(config)
    output_dir = config.checkpoint_dir / "anchor_memories" / f"task_{stage:02d}"

    logging.info(
        "Running anchor memory selection for stage %d (canonical task %d, reference=%s)",
        stage,
        next_canonical_task_id,
        reference_task_ids,
    )
    result = run_anchor_memory_pipeline(
        model,
        config,
        reference_task_ids=reference_task_ids,
        next_task_idx=next_canonical_task_id,
        output_dir=output_dir,
        anchor_cfg=anchor_cfg,
        norm_stats_root=norm_stats_root,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        save_artifacts=True,
    )

    assert len(result.buffer_lst) == replay_end == len(reference_task_ids)
    for i in range(replay_end):
        buffer_lst[i] = result.buffer_lst[i]

    logging.info(
        "Anchor buffers set for replay slice 0:%d (%d total selected frames)",
        replay_end,
        sum(len(buffer_lst[i]) for i in range(replay_end)),
    )


def _prepare_replay_buffer_for_task(
    config: _config.TrainConfig,
    train_state,
    *,
    stage: int,
    order: list[int],
    dataset_lst: list[Any],
    buffer_lst: list[list[int]],
) -> None:
    """Prepare replay buffer before training stage `stage` (anchor selection or post-boundary resample)."""
    if stage == 0:
        return

    if config.cl == "er_anchor":
        _run_anchor_selection_for_task(
            config,
            train_state,
            stage=stage,
            order=order,
            buffer_lst=buffer_lst,
        )
        return

    replay_end = _replay_task_slice_end(stage)
    if config.n_memories_total or config.n_memories_proportion is not None:
        pool_sizes = [len(dataset_lst[i]) for i in range(replay_end)]
        resample_replay_buffer_lst(
            config,
            pool_sizes,
            buffer_lst,
            replay_end,
            seed=config.seed,
        )


def _er_uses_mixed_batch(config: _config.TrainConfig, *, task_idx: int, replay_loader) -> bool:
    """True when training concatenates current-task and replay batches (ER from task 1 onward)."""
    return _uses_er(config) and task_idx > 0 and replay_loader is not None


def _log_er_mixed_batch(config: _config.TrainConfig, obs: _model.Observation, *, task_idx: int) -> None:
    """Debug log: verify mixed batch is 2×batch_size with distinct current vs replay prompts."""
    total_bs = _batch_size_from_obs(obs)
    per_loader_bs = config.batch_size
    logging.info(
        "ER mixed batch on task %d: total_batch_size=%d, per_loader_batch_size=%d",
        task_idx,
        total_bs,
        per_loader_bs,
    )
    if total_bs != 2 * per_loader_bs:
        logging.warning(
            "Expected balanced ER batch of size %d (current + replay), got %d",
            2 * per_loader_bs,
            total_bs,
        )
        return

    cur_sigs = _prompt_signatures(obs, 0, per_loader_bs)
    rep_sigs = _prompt_signatures(obs, per_loader_bs, total_bs)
    overlap = len(cur_sigs & rep_sigs)
    logging.info(
        "  current half: %d samples, %d unique prompts, examples=%s",
        per_loader_bs,
        len(cur_sigs),
        _decode_sample_prompts(obs, n=3, start=0),
    )
    logging.info(
        "  replay half: %d samples, %d unique prompts, examples=%s",
        per_loader_bs,
        len(rep_sigs),
        _decode_sample_prompts(obs, n=3, start=per_loader_bs),
    )
    logging.info("  prompt overlap between halves: %d shared (0 expected for distinct tasks)", overlap)


def _debug_inspect_er_loaders(
    config: _config.TrainConfig,
    *,
    task_idx: int,
    loader,
    replay_loader,
    mixed_loader,
) -> None:
    """Debug log: sample one batch from current, replay, and mixed loaders at task start."""
    current_repo = loader.data_config().repo_id
    logging.info(
        "ER debug task %d: current_repo=%s, replay_active=%s",
        task_idx,
        current_repo,
        replay_loader is not None and task_idx > 0,
    )

    cur_obs, _ = next(iter(loader))
    logging.info(
        "  current-task loader batch_size=%d, prompts=%s",
        _batch_size_from_obs(cur_obs),
        _decode_sample_prompts(cur_obs),
    )

    if replay_loader is not None and task_idx > 0:
        rep_obs, _ = next(iter(replay_loader))
        logging.info(
            "  replay loader batch_size=%d, prompts=%s",
            _batch_size_from_obs(rep_obs),
            _decode_sample_prompts(rep_obs),
        )

    mix_obs, _ = next(iter(mixed_loader))
    if _er_uses_mixed_batch(config, task_idx=task_idx, replay_loader=replay_loader):
        _log_er_mixed_batch(config, mix_obs, task_idx=task_idx)
    else:
        logging.info(
            "  training loader batch_size=%d (current-task only; ER mixing starts at task 1)",
            _batch_size_from_obs(mix_obs),
        )


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Load pretrained weights and return a pure dict matching trainable param shapes."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)

        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        chunked_loss = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss)

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    diff_state = nnx.DiffState(0, config.trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model, train_rng, observation, actions)

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    return new_state, {"loss": loss}


def make_single_task_loop(
    config,
    mesh,
    train_rng,
    train_state,
    train_state_sharding,
    data_sharding,
    replicated_sharding,
    data_loader,
    task_loaders,
    task_idx,
    checkpoint_manager,
    replay_loader=None,
    buffer_lst=None,
):
    """Train on one sequential task for `steps_per_task` steps; optionally checkpoint ER `buffer_lst`."""
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")
    if config.debug and _uses_er(config):
        if _er_uses_mixed_batch(config, task_idx=task_idx, replay_loader=replay_loader):
            _log_er_mixed_batch(config, batch[0], task_idx=task_idx)
        else:
            logging.info(
                "ER batch on task %d: batch_size=%d (current-task only; ER mixing starts at task 1)",
                task_idx,
                _batch_size_from_obs(batch[0]),
            )

    global_start = int(jax.device_get(train_state.step))

    if config.wandb_enabled:
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(5, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=global_start)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    global_end = global_start + config.steps_per_task

    pbar = tqdm.tqdm(
        range(global_start, global_end),
        initial=global_start,
        total=global_end,
        dynamic_ncols=True,
        desc=f"T{task_idx}",
    )

    infos = []
    for global_step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)

        jax.block_until_ready(train_state)
        infos.append(info)

        if global_step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {global_step}: {info_str}")
            if config.wandb_enabled:
                wandb.log(reduced_info, step=global_step)
            infos = []

        if (
            not config.debug
            and not config.save_end_of_task_only
            and global_step % config.save_interval == 0
            and global_step > 0
        ):
            _checkpoints.save_state(
                checkpoint_manager,
                train_state,
                data_loader,
                global_step,
                task_loaders,
                replay_loader=replay_loader,
                task_idx=task_idx,
                buffer_lst=buffer_lst,
            )

        batch = next(data_iter)

    if not config.debug:
        _checkpoints.save_state(
            checkpoint_manager,
            train_state,
            data_loader,
            global_step,
            task_loaders,
            replay_loader=replay_loader,
            task_idx=task_idx,
            buffer_lst=buffer_lst,
        )

    return train_state


def _prune_future_checkpoints(checkpoint_manager: ocp.CheckpointManager, keep_step: int):
    """On resume, delete checkpoints saved after `keep_step` (from a rolled-back run)."""
    checkpoint_manager.reload()
    for s in checkpoint_manager.all_steps(read=True):
        if s > keep_step:
            logging.info(f"Deleting future checkpoint at step {s}")
            checkpoint_manager.delete(s)


def main(config: _config.TrainConfig):
    init_logging()

    _validate_cl_mode(config)

    if config.debug:
        config = config.effective_sequential_settings()
        logging.info(
            "Debug mode enabled: steps_per_task=%d, n_tasks=%d, batch_size=%d, n_memories=%d, "
            "checkpoints=disabled",
            config.steps_per_task,
            config.n_tasks,
            config.batch_size,
            config.n_memories,
        )

    if config.no_resampling:
        raise ValueError(
            "train_sequential does not support no_resampling; "
            "current-task and replay batches are always mixed via MixedDataLoader."
        )

    logging.info(f"Running on: {platform.node()}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        max_to_keep=config.max_to_keep,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)

    n_tasks = config.n_tasks
    order = _resolve_training_task_order(config)
    _assert_task_order_matches_checkpoint(config.checkpoint_dir, order)
    if not resuming:
        _persist_task_order(
            config.checkpoint_dir,
            task_order_index=config.task_order_index,
            order=order,
        )
    logging.info(
        "Training task order (stage -> canonical LIBERO id): %s",
        [f"{stage}->{order[stage]}" for stage in range(n_tasks)],
    )

    logging.info("Creating per-stage dataloaders")
    task_loaders = []
    for stage in range(n_tasks):
        canonical_task_id = _canonical_task_id(order, stage)
        loader = _data_loader.create_data_loader_sequential(
            config,
            task_id=canonical_task_id,
            shuffle=True,
            sharding=data_sharding,
            **(
                {"cust_batch_size": int(config.batch_size * 2)}
                if stage == 0 and config.double_batch_start
                else {}
            ),
        )
        _assert_loader_matches_canonical_task(
            config,
            loader,
            stage=stage,
            canonical_task_id=canonical_task_id,
        )
        task_loaders.append(loader)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    prev_task_idx = -1
    replay_loader = None

    if config.cl == "er":
        dataset_lst, buffer_lst = create_deterministic_buffer(
            config,
            n_tasks=n_tasks,
            shuffle=True,
            sharding=data_sharding,
            task_order=order,
        )
        assert len(dataset_lst) == len(buffer_lst) == n_tasks
        if config.debug:
            _log_er_replay_buffer(config, buffer_lst=buffer_lst, task_order=order)
    elif config.cl == "er_anchor":
        full_dataset_lst, _ = create_full_task_datasets(
            config,
            n_tasks=n_tasks,
            task_order=order,
        )
        dataset_lst = full_dataset_lst
        buffer_lst = [[] for _ in range(n_tasks)]
        assert len(dataset_lst) == n_tasks
        if config.debug:
            logging.info("er_anchor: full datasets loaded (%d tasks), buffers filled at each task boundary", n_tasks)
    else:
        dataset_lst, buffer_lst = [], []
        full_dataset_lst = []

    if resuming:
        if _uses_er(config):
            step_to_restore = _checkpoints.latest_task_checkpoint_step(checkpoint_manager)
            logging.info(f"Restoring from step: {step_to_restore}")

            train_state, prev_task_idx, saved_buffer_lst = _checkpoints.restore_state(
                checkpoint_manager,
                train_state,
                data_loader=None,
                step=step_to_restore,
                restore_task_idx=True,
                restore_replay_buffer=True,
            )
            _prune_future_checkpoints(checkpoint_manager, step_to_restore)

            if config.cl == "er":
                dataset_lst, buffer_lst = create_deterministic_buffer(
                    config,
                    n_tasks=n_tasks,
                    shuffle=True,
                    sharding=data_sharding,
                    task_order=order,
                    preset_buffers=saved_buffer_lst,
                )
            else:
                full_dataset_lst, _ = create_full_task_datasets(config, n_tasks=n_tasks, task_order=order)
                dataset_lst = full_dataset_lst
                buffer_lst = saved_buffer_lst

            next_stage = prev_task_idx + 1
            _prepare_replay_buffer_for_task(
                config,
                train_state,
                stage=next_stage,
                order=order,
                dataset_lst=dataset_lst,
                buffer_lst=buffer_lst,
            )
            replay_end = _replay_task_slice_end(next_stage)
            replay_loader = _build_replay_loader(
                config,
                dataset_lst,
                buffer_lst,
                replay_end=replay_end,
                data_config=task_loaders[next_stage].data_config(),
                data_sharding=data_sharding,
            )
            if config.debug and replay_loader is not None:
                subsets = build_replay_subsets(dataset_lst[:replay_end], buffer_lst[:replay_end])
                _log_er_replay_loader(
                    config,
                    dataset_lst=subsets,
                    replay_task_indices=_replay_task_ids(order, replay_end),
                )
        else:
            train_state, prev_task_idx, _ = _checkpoints.restore_state(
                checkpoint_manager,
                train_state,
                data_loader=None,
                step=checkpoint_manager.latest_step(),
                restore_task_idx=True,
            )

        logging.info(f"Resuming from task {prev_task_idx}")

    def _run_task(stage: int):
        nonlocal train_state, replay_loader

        canonical_task_id = _canonical_task_id(order, stage)
        loader = task_loaders[stage]
        short_repo = _task_repo_id(config, canonical_task_id).rsplit("/", 1)[-1]
        logging.info(
            "=== Learning stage %d / %d (canonical task %d: %s) ===",
            stage,
            n_tasks - 1,
            canonical_task_id,
            short_repo,
        )
        _assert_loader_matches_canonical_task(
            config,
            loader,
            stage=stage,
            canonical_task_id=canonical_task_id,
        )

        if _uses_er(config):
            if config.cl == "er_anchor" and stage > 0:
                _prepare_replay_buffer_for_task(
                    config,
                    train_state,
                    stage=stage,
                    order=order,
                    dataset_lst=dataset_lst,
                    buffer_lst=buffer_lst,
                )
                replay_end = _replay_task_slice_end(stage)
                replay_loader = _build_replay_loader(
                    config,
                    dataset_lst,
                    buffer_lst,
                    replay_end=replay_end,
                    data_config=loader.data_config(),
                    data_sharding=data_sharding,
                )
                if config.debug and replay_loader is not None:
                    subsets = build_replay_subsets(dataset_lst[:replay_end], buffer_lst[:replay_end])
                    _log_er_replay_loader(
                        config,
                        dataset_lst=subsets,
                        replay_task_indices=_replay_task_ids(order, replay_end),
                    )

            mixed_loader = (
                MixedDataLoader(loader.data_config(), loader, replay_loader) if stage > 0 else loader
            )
        else:
            mixed_loader = loader

        if config.debug and _uses_er(config):
            _debug_inspect_er_loaders(
                config,
                task_idx=stage,
                loader=loader,
                replay_loader=replay_loader,
                mixed_loader=mixed_loader,
            )

        train_state = make_single_task_loop(
            config,
            mesh,
            train_rng,
            train_state,
            train_state_sharding,
            data_sharding,
            replicated_sharding,
            mixed_loader,
            task_loaders,
            stage,
            checkpoint_manager,
            replay_loader=replay_loader,
            buffer_lst=buffer_lst if _uses_er(config) else None,
        )

        if config.cl == "er":
            next_stage = stage + 1
            _prepare_replay_buffer_for_task(
                config,
                train_state,
                stage=next_stage,
                order=order,
                dataset_lst=dataset_lst,
                buffer_lst=buffer_lst,
            )
            replay_end = _replay_task_slice_end(next_stage)
            if next_stage < n_tasks:
                replay_loader = _build_replay_loader(
                    config,
                    dataset_lst,
                    buffer_lst,
                    replay_end=replay_end,
                    data_config=task_loaders[next_stage].data_config(),
                    data_sharding=data_sharding,
                )
            if config.debug and replay_loader is not None:
                subsets = build_replay_subsets(dataset_lst[:replay_end], buffer_lst[:replay_end])
                _log_er_replay_loader(
                    config,
                    dataset_lst=subsets,
                    replay_task_indices=_replay_task_ids(order, replay_end),
                )

    if config.learn_one_task_idx > -1:
        _run_task(config.learn_one_task_idx)
    else:
        for stage in range(prev_task_idx + 1, n_tasks):
            _run_task(stage)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
