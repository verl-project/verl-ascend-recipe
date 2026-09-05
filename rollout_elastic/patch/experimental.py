# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Patch ``verl.experimental`` for elastic-rollout fault tolerance.

This area wires the recipe-owned ``fault_tolerance`` package into the
experimental (async) trainer families without shipping modified copies of
verl's experimental sources:

- ``agent_loop``: ``AgentLoopWorker`` / ``AgentLoopManager`` learn to drop
  faulted sub-batches (``filter_partial_batch``) instead of crashing the
  whole generation call.
- ``separation.ray_trainer``: ``SeparateRayPPOTrainer`` builds a
  ``ThreadedSupervisor`` in ``init_workers`` and feeds the recipe's
  ``fault_tolerance`` / ``load_balancer_handle`` into the checkpoint manager.
- ``one_step_off_policy.ray_trainer``: ``OneStepOffRayTrainer`` enables the
  retrying LLM client and token-continuation store, pads/truncates batches
  after partial generation, and starts/stops the Supervisor around training.
- ``fully_async_policy``: ``FullyAsyncRollouter`` / ``FullyAsyncTrainer`` get
  the Supervisor, cross-actor death/promotion callbacks and token continuation;
  ``FullyAsyncTaskRunner`` wires the Supervisor to the trainer's CKE.

Everything is expressed with ``@patch`` / ``@add`` / ``@wrap``; no verl source
file is modified on disk.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from pprint import pprint

import ray
from omegaconf import OmegaConf

from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.experimental.fully_async_policy.detach_utils import safe_create_task
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter,
)
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.protocol import DataProto
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import RolloutTraceConfig
from verl.utils.tracking import Tracking
from verl.workers.rollout.fault_tolerance import filter_partial_batch
from verl.workers.rollout.llm_server import LLMServerManager

from ._core import add, patch, wrap

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# agent_loop — AgentLoopWorker
# ---------------------------------------------------------------------------
@patch(AgentLoopWorker, "generate_sequences")
async def _worker_generate_sequences(self, batch: DataProto) -> DataProto:
    """Generate sequences from agent loop (FT-tolerant partial batch).

    When fault tolerance is enabled, prompt-level faults are absorbed and the
    surviving samples are kept, instead of failing the whole batch.
    """
    config = self.rollout_config
    sampling_params = dict(
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        repetition_penalty=1.0,
        logprobs=config.calculate_log_probs,
    )

    # override sampling params for validation
    if batch.meta_info.get("validate", False):
        sampling_params["top_p"] = config.val_kwargs.top_p
        sampling_params["top_k"] = config.val_kwargs.top_k
        sampling_params["temperature"] = config.val_kwargs.temperature

    # by default, we assume it's a single turn agent
    if "agent_name" not in batch.non_tensor_batch:
        default_agent_loop = config.agent.default_agent_loop
        batch.non_tensor_batch["agent_name"] = __import__("numpy").array(
            [default_agent_loop] * len(batch), dtype=object
        )

    if "index" in batch.non_tensor_batch:
        index = batch.non_tensor_batch["index"]
    else:
        index = __import__("numpy").arange(len(batch))

    max_samples_per_worker = RolloutTraceConfig.get_instance().max_samples_per_step_per_worker

    # For n rollouts per sample, we trace all n rollouts for selected samples
    # Note: This sampling happens per-worker, so total traces = max_samples_per_worker * num_workers * n
    if max_samples_per_worker is not None:
        unique_sample_indices = __import__("numpy").unique(index)
        if max_samples_per_worker < len(unique_sample_indices):
            selected_samples = set(
                __import__("numpy").random.choice(unique_sample_indices, max_samples_per_worker, replace=False).tolist()
            )
            traced_indices = set(i for i in range(len(batch)) if index[i] in selected_samples)
        else:
            traced_indices = set(range(len(batch)))
    else:
        traced_indices = set(range(len(batch)))

    trajectory_info = await get_trajectory_info(
        batch.meta_info.get("global_steps", -1), index.tolist(), batch.meta_info.get("validate", False)
    )

    tasks = []
    for i in range(len(batch)):
        trace_this_sample = i in traced_indices
        kwargs = {k: v[i] for k, v in batch.non_tensor_batch.items()}
        tasks.append(
            asyncio.create_task(
                self._run_agent_loop(sampling_params, trajectory_info[i], trace=trace_this_sample, **kwargs)
            )
        )

    if self._ft_enabled():
        results = await asyncio.gather(*tasks, return_exceptions=True)
        outputs, ok_indices = filter_partial_batch(results, self._ft_min_ok_ratio())
        input_non_tensor_batch = {k: v[ok_indices] for k, v in batch.non_tensor_batch.items()}
    else:
        outputs = await asyncio.gather(*tasks)
        input_non_tensor_batch = batch.non_tensor_batch

    if not outputs:
        return DataProto()

    output = self._postprocess(
        outputs, input_non_tensor_batch=input_non_tensor_batch, validate=batch.meta_info.get("validate", False)
    )
    return output


@add(AgentLoopWorker, "_ft_enabled")
def _worker_ft_enabled(self) -> bool:
    return bool(OmegaConf.select(self.config, "async_training.fault_tolerance.enabled", default=False))


@add(AgentLoopWorker, "_ft_min_ok_ratio")
def _worker_ft_min_ok_ratio(self) -> float:
    return float(OmegaConf.select(self.config, "async_training.fault_tolerance.min_ok_ratio", default=0.5))


# ---------------------------------------------------------------------------
# agent_loop — AgentLoopManager
# ---------------------------------------------------------------------------
@add(AgentLoopManager, "_mgr_ft_enabled")
def _mgr_ft_enabled(self) -> bool:
    return bool(OmegaConf.select(self.config, "async_training.fault_tolerance.enabled", default=False))


@patch(AgentLoopManager, "generate_sequences")
async def _mgr_generate_sequences(self, prompts: DataProto) -> DataProto:
    """Split input batch and dispatch to agent loop workers (FT-tolerant)."""
    chunkes = prompts.chunk(len(self.agent_loop_workers))
    ft_on = self._mgr_ft_enabled()
    worker_futures = [
        worker.generate_sequences.remote(chunk) for worker, chunk in zip(self.agent_loop_workers, chunkes, strict=True)
    ]
    if ft_on:
        results = await asyncio.gather(*worker_futures, return_exceptions=True)
        outputs = []
        for r in results:
            if isinstance(r, BaseException):
                logger.warning("[FT] AgentLoopManager: dropping worker batch due to %r", r)
                continue
            outputs.append(r)
        if not outputs:
            first = results[0] if results else None
            if isinstance(first, BaseException):
                raise RuntimeError(f"[FT] all agent loop workers failed, first error: {first}") from first
            raise RuntimeError("[FT] all agent loop workers failed; nothing to concat")
    else:
        outputs = await asyncio.gather(*worker_futures)
    output = DataProto.concat(outputs)

    # calculate performance metrics
    metrics = [output.meta_info.pop("metrics") for output in outputs]  # List[List[Dict[str, str]]]
    timing = self._performance_metrics(metrics, output)

    output.meta_info = {"timing": timing, **outputs[0].meta_info}
    return output


# ---------------------------------------------------------------------------
# separation.ray_trainer — SeparateRayPPOTrainer
# ---------------------------------------------------------------------------
@wrap(SeparateRayPPOTrainer, "__init__")
def _sep_trainer_init(orig, self, *args, **kwargs):
    orig(self, *args, **kwargs)
    self._ft_probe_rollout_replica = None


@patch(SeparateRayPPOTrainer, "init_workers")
def _sep_trainer_init_workers(self):
    """Initialize distributed training workers using Ray backend.

    Creates:
    1. Ray resource pools from configuration
    2. Worker groups for each role (actor, critic, etc.)
    3. The FT ``ThreadedSupervisor`` (when enabled) wired to the checkpoint manager.
    """
    self._init_resource_pools()
    self._create_worker_classes()
    self._init_worker_groups()
    self._init_models()
    self._init_reward_loop()
    self._init_async_rollout_manager()

    # Support custom CheckpointEngineManager via config
    checkpoint_manager_class_fqn = self.config.actor_rollout_ref.rollout.get("checkpoint_manager_class")
    if checkpoint_manager_class_fqn:
        from verl.utils.import_utils import load_class_from_fqn

        CheckpointEngineManager = load_class_from_fqn(checkpoint_manager_class_fqn, "CheckpointEngineManager")
    else:
        from verl.checkpoint_engine import CheckpointEngineManager

    ft_cfg = None
    try:
        from verl.workers.rollout.fault_tolerance import FaultToleranceConfig

        ft_node = OmegaConf.select(self.config, "async_training.fault_tolerance")
        if ft_node is not None:
            ft_cfg = FaultToleranceConfig(**OmegaConf.to_container(ft_node, resolve=True))
    except Exception:
        ft_cfg = None

    from verl.utils.config import omega_conf_to_dataclass

    self.checkpoint_manager = CheckpointEngineManager(
        config=omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine),
        trainer=self.actor_rollout_wg,
        replicas=self.llm_server_manager.get_replicas(),
        fault_tolerance=ft_cfg,
        load_balancer_handle=self.llm_server_manager.global_load_balancer,
    )
    self._ft_supervisor = None
    if ft_cfg is not None and ft_cfg.enabled:
        from verl.workers.rollout.fault_tolerance import (
            Supervisor,
            ThreadedSupervisor,
            make_on_dead,
        )

        async def probe_fn(replica):
            try:
                return bool(await asyncio.wait_for(replica.health(), timeout=2.0))
            except Exception:
                return False

        replicas = self.llm_server_manager.get_replicas()
        # replica_id = _server_address (matches LB's server_id keying)
        replica_map = {r._server_address: r for r in replicas}

        spawner_fn = None
        on_spawn_success_fn = None
        if getattr(ft_cfg, "replace_dead_replicas", False):
            _ft_log = logging.getLogger(__name__)

            async def spawner_fn(dead_id):  # noqa: E306
                return await self.llm_server_manager.spawn_replacement(dead_id)

            async def on_spawn_success_fn(dead_id, new_replica):  # noqa: E306
                # Keep a replacement out of the current sync/LB until a
                # complete weight transaction successfully commits it.
                self.checkpoint_manager.add_pending_replicas([new_replica])
                sup = getattr(self, "_ft_supervisor", None)
                if sup is not None:
                    sup.supervisor.add_replica(new_replica._server_address, new_replica)
                _ft_log.warning(
                    "[FT] on_spawn_success: replica %s added back",
                    new_replica._server_address,
                )

        on_dead_handler = make_on_dead(
            lb_handle=self.llm_server_manager.global_load_balancer,
            replica_to_server_ids=lambda rid: [rid],
            ckpt_mgr_callback=self.checkpoint_manager.on_replica_dead,
            spawner=spawner_fn,
            on_spawn_success=on_spawn_success_fn,
        )

        async def promote_fn(servers):
            await self.llm_server_manager.global_load_balancer.add_servers.remote(servers)

        inner_sup = Supervisor(
            replicas=replica_map,
            probe_fn=probe_fn,
            on_dead=on_dead_handler,
            promote_fn=promote_fn,
            interval_s=ft_cfg.heartbeat_interval_s,
            miss_threshold=ft_cfg.heartbeat_miss_threshold,
            probe_timeout_s=2.0,
        )
        # Own thread+loop so trainer-main blocking ray.get can't starve heartbeat.
        self._ft_supervisor = ThreadedSupervisor(inner_sup)
        self.checkpoint_manager.set_sync_failure_reporter(self._ft_supervisor.report_failure)
        self.checkpoint_manager.set_replica_promotion_reporter(self._ft_supervisor.promote_replica)
        logging.getLogger(__name__).warning(
            "[FT] init_workers: ThreadedSupervisor created with %d replicas, interval=%s miss_threshold=%s",
            len(replica_map),
            ft_cfg.heartbeat_interval_s,
            ft_cfg.heartbeat_miss_threshold,
        )


# ---------------------------------------------------------------------------
# one_step_off_policy.ray_trainer — OneStepOffRayTrainer
# ---------------------------------------------------------------------------
@patch(OneStepOffRayTrainer, "_init_async_rollout_manager")
def _one_step_init_async_rollout_manager(self):
    # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
    # agent_reward_loop: streaming reward computation with actor rollout
    # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
    enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

    # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
    # to stream reward computation with actor rollout
    reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

    # create async rollout manager and request scheduler
    assert self.config.actor_rollout_ref.rollout.mode == "async"

    # Support custom AgentLoopManager via config
    manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
    if manager_class_fqn:
        from verl.utils.import_utils import load_class_from_fqn

        AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
    else:
        from verl.experimental.agent_loop import AgentLoopManager

    self.llm_server_manager = LLMServerManager.create(config=self.config)
    self._init_one_step_progress()
    self.async_rollout_mode = True
    self.async_rollout_manager = AgentLoopManager.create(
        config=self.config,
        llm_client=self.llm_server_manager.get_client(retry=True),
        reward_loop_worker_handles=reward_loop_worker_handles,
    )


@add(OneStepOffRayTrainer, "_init_one_step_progress")
@auto_await
async def _one_step_init_progress(self) -> None:
    import logging as _ft_logging

    try:
        ft_enabled = bool(OmegaConf.select(self.config, "async_training.fault_tolerance.enabled", default=False))
        progress_enabled = bool(
            OmegaConf.select(self.config, "async_training.fault_tolerance.progress.enabled", default=False)
        )
    except Exception as e:
        _ft_logging.getLogger(__name__).warning("[FT] init_one_step_progress failed: %s", e)
        return
    if not ft_enabled or not progress_enabled:
        return

    progress_node = OmegaConf.select(self.config, "async_training.fault_tolerance.progress")
    progress_config = self._build_progress_config(progress_node)

    await self.llm_server_manager._init_progress_store(progress_config)
    _ft_logging.getLogger(__name__).info(
        "[FT] one-step token continuation enabled: run_id=%s, store=%s(client resolved via get_client(retry = True))",
        self.llm_server_manager.run_id,
        progress_config.persist_root,
    )


@add(OneStepOffRayTrainer, "_build_progress_config")
def _one_step_build_progress_config(self, progress_node):
    from verl.workers.rollout.fault_tolerance import ModelVersionPolicy, ProgressConfig

    if progress_node is None:
        return ProgressConfig()
    kwargs = {}
    for key, value in OmegaConf.to_container(progress_node, resolve=True).items():
        if key == "model_version_policy" and isinstance(value, dict):
            kwargs[key] = ModelVersionPolicy(mode=value.get("mode", "exact"))
        else:
            kwargs[key] = value
    return ProgressConfig(**kwargs)


@patch(OneStepOffRayTrainer, "_async_gen_next_batch")
async def _one_step_async_gen_next_batch(self, continuous_iterator):
    """Call parameter synchronization and asynchronous sequence generation."""
    import numpy as np
    import torch

    from verl.protocol import DataProto
    from verl.trainer.ppo.ray_trainer import compute_response_mask
    from verl.utils.debug import marked_timer

    try:
        epoch, batch_dict = next(continuous_iterator)
    except StopIteration:
        return None
    except Exception as e:
        print(f"Error in async_gen_next_batch: {e}")
        return None

    metrics = {}
    timing_raw = {}

    # Create the initial batch from the data loader
    batch = DataProto.from_single_dict(batch_dict)

    # add uid to batch
    batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)

    gen_batch = self._get_gen_batch(batch)

    # pass global_steps to trace
    gen_batch.meta_info["global_steps"] = self.global_steps
    gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

    # async generation
    with marked_timer("generate_async", timing_raw, color="purple"):
        gen_batch_output = await self.async_rollout_manager.generate_sequences(gen_batch_output)

    # repeat to align with repeated responses in rollout
    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
    if len(gen_batch_output) < len(batch):
        batch = batch[: len(gen_batch_output)]
    batch = batch.union(gen_batch_output)

    if "response_mask" not in batch.batch.keys():
        batch.batch["response_mask"] = compute_response_mask(batch)
    # Balance the number of valid tokens across DP ranks.
    # NOTE: This usually changes the order of data in the `batch`,
    # which won't affect the advantage calculation (since it's based on uid),
    # but might affect the loss calculation (due to the change of mini-batching).
    if self.config.trainer.balance_batch:
        self._balance_batch(batch, metrics=metrics)

    # compute global_valid tokens
    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

    # Launch individual reward computations as each generation completes
    future_reward = None

    # Return the original, now-modified `batch` and the `future_reward`
    return metrics, timing_raw, epoch, batch, future_reward


@patch(OneStepOffRayTrainer, "fit")
async def _one_step_fit(self):
    """The training loop of PPO (FT-aware Supervisor lifecycle)."""
    from tqdm import tqdm

    from verl.utils.rollout_skip import RolloutSkip

    self.logger = Tracking(
        project_name=self.config.trainer.project_name,
        experiment_name=self.config.trainer.experiment_name,
        default_backend=self.config.trainer.logger,
        config=OmegaConf.to_container(self.config, resolve=True),
    )

    self.global_steps = 0

    # Load the checkpoint before the first sync. Supervisor must already
    # be alive so a CKE-first rollout failure can be reported immediately.
    self._load_checkpoint()
    if getattr(self, "_ft_supervisor", None) is not None:
        import logging as _ft_logging

        _ft_logging.getLogger(__name__).warning("[FT] fit: starting Supervisor heartbeat before initial sync")
        self._ft_supervisor.start()
    else:
        import logging as _ft_logging

        _ft_logging.getLogger(__name__).warning("[FT] fit: _ft_supervisor is None — no FT detection")

    try:
        self._fit_update_weights()
    except BaseException:
        if getattr(self, "_ft_supervisor", None) is not None:
            self._ft_supervisor.stop()
        raise

    # perform validation before training
    # currently, we only support validation using the reward_function.
    if self.config.trainer.get("val_before_train", True):
        val_metrics = self._validate()
        assert val_metrics, f"{val_metrics=}"
        pprint(f"Initial validation metrics: {val_metrics}")
        self.logger.log(data=val_metrics, step=self.global_steps)
        if self.config.trainer.get("val_only", False):
            return

    if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
        rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
        rollout_skip.wrap_generate_sequences()

    # add tqdm
    self.progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

    # we start from step 1
    self.global_steps += 1
    self.last_val_metrics = None
    self.max_steps_duration = 0

    self.prev_step_profile = False
    self.curr_step_profile = (
        self.global_steps in self.config.global_profiler.steps
        if self.config.global_profiler.steps is not None
        else False
    )
    self.next_step_profile = False

    # across epoch iterator
    continuous_iterator = self._create_continuous_iterator()
    # Start the first asynchronous generation task.
    batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
    try:
        while batch_data_future is not None:
            batch_data_future = await self.fit_step(batch_data_future, continuous_iterator)
            if self.is_last_step:
                return
    finally:
        if getattr(self, "_ft_supervisor", None) is not None:
            # ThreadedSupervisor.stop() is synchronous — it bridges into its
            # own asyncio loop via run_coroutine_threadsafe internally.
            self._ft_supervisor.stop()


@patch(OneStepOffRayTrainer, "fit_step")
async def _one_step_fit_step(self, batch_data_future, continuous_iterator):
    """Single-step training template method (batch padded after partial gen)."""
    from verl.utils.debug import marked_timer

    self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
    self.timing_raw = {}
    # reward message
    self.future_reward = None
    self.reward_tensor = None
    self.reward_extra_infos_dict = {}

    self._fit_prepare_step()
    self._fit_start_profile()

    def pad_batch_to_size(batch, target_size):
        current_size = len(batch)
        if current_size == 0:
            print("Warning: empty batch returned from generation; skipping padding")
            return batch
        if current_size == target_size:
            return batch
        elif current_size > target_size:
            return batch[:target_size]
        else:
            print(f"Warning: Batch padded from {current_size} to {target_size}")
            repeats = target_size // current_size
            remainder = target_size % current_size

            repeated_parts = [batch] * repeats
            if remainder != 0:
                repeated_parts.append(batch[:remainder])

            padded_batch = DataProto.concat(repeated_parts)
            return padded_batch

    with marked_timer("step", self.timing_raw):
        batch, batch_data_future = await self._fit_generate(batch_data_future, continuous_iterator)

        batch = pad_batch_to_size(batch, self.config.actor_rollout_ref.rollout.n * self.config.data.train_batch_size)

        # await asyncio.sleep(0) ensures:
        # Asynchronous tasks can start executing immediately
        # The event loop can handle other pending coroutines
        # Prevents computations in a certain phase from blocking the entire asynchronous workflow
        #
        # The purpose here is to ensure that after triggering
        # `self.async_rollout_manager.generate_sequences(gen_batch_output)`,
        # the subsequent relevant logic can proceed in a timely manner
        await asyncio.sleep(0)
        batch = self._fit_compute_reward(batch)
        await asyncio.sleep(0)
        batch = self._fit_compute_log_prob(batch)
        await asyncio.sleep(0)
        batch = self._fit_compute_ref_log_prob(batch)
        await asyncio.sleep(0)
        batch = self._fit_compute_critic(batch)
        await asyncio.sleep(0)
        batch = self._fit_compute_advantage(batch)
        await asyncio.sleep(0)
        batch = self._fit_update_critic(batch)
        await asyncio.sleep(0)
        batch = self._fit_update_actor(batch)
        await asyncio.sleep(0)
        self._fit_dump_data(batch)
        await asyncio.sleep(0)

    self._fit_validate()
    await asyncio.sleep(0)
    self._fit_save_checkpoint()
    await asyncio.sleep(0)
    self._fit_stop_profile()
    self._fit_collect_metrics(batch)
    self._fit_experimental(batch)
    self._fit_postprocess_step()

    return batch_data_future


# ---------------------------------------------------------------------------
# fully_async_policy.fully_async_rollouter — FullyAsyncRollouter
# ---------------------------------------------------------------------------
@wrap(FullyAsyncRollouter, "__init__")
def _rollouter_init(orig, self, *args, **kwargs):
    orig(self, *args, **kwargs)
    # Fault tolerance is constructed before the trainer's first sync and
    # started by init_ft_supervisor so CKE-first failures are reportable.
    self._ft_supervisor = None
    self._trainer_handle = None


@add(FullyAsyncRollouter, "get_load_balancer")
def _rollouter_get_load_balancer(self):
    """Get the global load balancer for FT-enabled checkpoint manager construction."""
    return self.llm_server_manager.global_load_balancer


@add(FullyAsyncRollouter, "init_ft_supervisor")
async def _rollouter_init_ft_supervisor(self, trainer_handle):
    """Construct the ThreadedSupervisor for inference instance elasticity.

    Called after the trainer's CKE is set up (via set_rollouter), so the
    cross-actor callbacks can safely reach the trainer's checkpoint_manager.

    The Supervisor lives in the Rollouter because it owns the replicas +
    LB + spawn_replacement. CKE membership notifications cross the actor
    boundary via Ray RPC to the trainer.

    Args:
        trainer_handle: Ray actor handle for FullyAsyncTrainer, used for
            cross-actor CKE membership notifications (on_replica_dead /
            on_replica_added).
    """
    import logging as _ft_logging

    self._trainer_handle = trainer_handle
    _ft_log = _ft_logging.getLogger(__name__)

    ft_cfg = None
    try:
        from verl.workers.rollout.fault_tolerance import FaultToleranceConfig

        ft_node = OmegaConf.select(self.config, "async_training.fault_tolerance")
        if ft_node is not None:
            ft_cfg = FaultToleranceConfig(**OmegaConf.to_container(ft_node, resolve=True))
    except Exception:
        ft_cfg = None

    self._ft_supervisor = None
    if ft_cfg is None or not ft_cfg.enabled:
        _ft_log.warning("[FT] init_ft_supervisor: fault_tolerance not enabled, skipping Supervisor")
        return

    from verl.workers.rollout.fault_tolerance import (
        Supervisor,
        ThreadedSupervisor,
        make_on_dead,
    )

    async def probe_fn(replica):
        try:
            return bool(await asyncio.wait_for(replica.health(), timeout=2.0))
        except Exception:
            return False

    replicas = self.llm_server_manager.get_replicas()
    # replica_id = _server_address (matches LB's server_id keying)
    replica_map = {r._server_address: r for r in replicas}

    # Cross-actor callback: notify Trainer so its CKE prunes the dead
    # replica. The Manager owns all communication-group reset decisions.
    async def ckpt_mgr_callback(replica_id):
        if self._trainer_handle is None:
            return
        try:
            ref = self._trainer_handle._on_replica_dead_from_supervisor.remote(replica_id)
            await asyncio.wrap_future(ref.future())
        except Exception as e:
            _ft_log.warning(
                "[FT] ckpt_mgr_callback: failed to notify trainer of replica death %s: %s",
                replica_id,
                e,
            )

    spawner_fn = None
    on_spawn_success_fn = None
    if getattr(ft_cfg, "replace_dead_replicas", False):

        async def spawner_fn(dead_id):  # noqa: E306
            return await self.llm_server_manager.spawn_replacement(dead_id)

        async def on_spawn_success_fn(dead_id, new_replica):  # noqa: E306
            # Register as pending first; Manager promotes it after a full
            # sync at the current target version, then LB admission is safe.
            if self._trainer_handle is None:
                raise RuntimeError("trainer handle is unavailable while registering a replacement replica")
            ref = self._trainer_handle._on_replica_added_from_supervisor.remote(new_replica)
            await asyncio.wrap_future(ref.future())
            sup = getattr(self, "_ft_supervisor", None)
            if sup is not None:
                sup.supervisor.add_replica(new_replica._server_address, new_replica)
            _ft_log.warning(
                "[FT] on_spawn_success: replica %s added back",
                new_replica._server_address,
            )

    on_dead_handler = make_on_dead(
        lb_handle=self.llm_server_manager.global_load_balancer,
        replica_to_server_ids=lambda rid: [rid],
        ckpt_mgr_callback=ckpt_mgr_callback,
        spawner=spawner_fn,
        on_spawn_success=on_spawn_success_fn,
    )

    async def promote_fn(servers):
        await self.llm_server_manager.global_load_balancer.add_servers.remote(servers)

    inner_sup = Supervisor(
        replicas=replica_map,
        probe_fn=probe_fn,
        on_dead=on_dead_handler,
        promote_fn=promote_fn,
        interval_s=ft_cfg.heartbeat_interval_s,
        miss_threshold=ft_cfg.heartbeat_miss_threshold,
        probe_timeout_s=2.0,
    )
    # Own thread+loop so rollouter's blocking operations can't starve heartbeat.
    self._ft_supervisor = ThreadedSupervisor(inner_sup)
    # The fully-async main performs its initial parameter sync immediately
    # after this method returns, so heartbeat/reporting must already run.
    self._ft_supervisor.start()
    _ft_log.warning(
        "[FT] init_ft_supervisor: ThreadedSupervisor started with %d replicas, interval=%s miss_threshold=%s",
        len(replica_map),
        ft_cfg.heartbeat_interval_s,
        ft_cfg.heartbeat_miss_threshold,
    )


@add(FullyAsyncRollouter, "report_sync_failure")
def _rollouter_report_sync_failure(self, replica_id: str, source: str = "unknown") -> None:
    """Forward a CKE sync failure to the rollouter-owned Supervisor."""
    supervisor = getattr(self, "_ft_supervisor", None)
    if supervisor is None:
        return
    supervisor.report_failure(replica_id, source)


@add(FullyAsyncRollouter, "promote_synced_replica")
async def _rollouter_promote_synced_replica(
    self,
    replica_id: str,
    servers: dict,
    attempt_id: int,
    target_version: int,
) -> bool:
    """Serialize serving admission with Supervisor death handling."""
    supervisor = getattr(self, "_ft_supervisor", None)
    if supervisor is None:
        return False
    return await supervisor.promote_replica(
        replica_id,
        servers,
        attempt_id,
        target_version,
    )


@patch(FullyAsyncRollouter, "_init_async_rollout_manager")
async def _rollouter_init_async_rollout_manager(self):
    # infrastructure overview: https://verl.readthedocs.io/en/latest/advance/reward_loop.html#architecture-design
    # agent_reward_loop: streaming reward computation with actor rollout
    # two conditions satisfied: (1) no reward model, or (2) reward model with extra resource pool
    enable_agent_reward_loop = not self.use_rm or self.config.reward.reward_model.enable_resource_pool

    # if enable_agent_reward_loop, we directly pass reward_loop_workers to agent loop manager
    # to stream reward computation with actor rollout
    reward_loop_worker_handles = self.reward_loop_manager.reward_loop_workers if enable_agent_reward_loop else None

    # create async rollout manager and request scheduler
    assert self.config.actor_rollout_ref.rollout.mode == "async"

    self.async_rollout_mode = True
    self.llm_server_manager = await LLMServerManager.create(config=self.config)
    await self._init_fully_async_progress()
    self.async_rollout_manager = await FullyAsyncAgentLoopManager.create(
        config=self.config,
        llm_client=self.llm_server_manager.get_client(fully_async=True),
        reward_loop_worker_handles=reward_loop_worker_handles,
        teacher_client=self.teacher_model_manager.get_client() if self.teacher_model_manager else None,
    )


@add(FullyAsyncRollouter, "_init_fully_async_progress")
async def _rollouter_init_fully_async_progress(self):
    """Mode C: create + initialise the RolloutProgressStoreActor if enabled.

    Mirrors one_step_off_policy's _init_one_step_progress. Only runs when both
    ``async_training.fault_tolerance.enabled`` and
    ``async_training.fault_tolerance.progress.enabled`` are True. The store handle
    is wired into every FullyLLMServerClient produced by
    ``llm_server_manager.get_client(fully_async=True)``, which is what turns on the
    token-continuation retry path.
    """
    import logging as _ft_logging

    try:
        ft_enabled = bool(OmegaConf.select(self.config, "async_training.fault_tolerance.enabled", default=False))
        progress_enabled = bool(
            OmegaConf.select(self.config, "async_training.fault_tolerance.progress.enabled", default=False)
        )
    except Exception as e:
        _ft_logging.getLogger(__name__).warning("[FT] init fully-async progress failed: %s", e)
        return
    if not ft_enabled or not progress_enabled:
        _ft_logging.getLogger(__name__).info(
            "[FT] fully-async token continuation skipped (ft.enabled=%s, progress.enabled=%s)",
            ft_enabled,
            progress_enabled,
        )
        return

    progress_node = OmegaConf.select(self.config, "async_training.fault_tolerance.progress")
    progress_config = self._build_progress_config(progress_node)
    await self.llm_server_manager._init_progress_store(progress_config)
    _ft_logging.getLogger(__name__).info(
        "[FT] fully-async Mode C (token continuation) enabled: run_id=%s, persist_root=%s",
        self.llm_server_manager.run_id,
        progress_config.persist_root,
    )


@add(FullyAsyncRollouter, "_build_progress_config")
def _rollouter_build_progress_config(self, progress_node):
    """Map the config ``progress`` node onto a ProgressConfig dataclass."""
    from verl.workers.rollout.fault_tolerance import ModelVersionPolicy, ProgressConfig

    if progress_node is None:
        return ProgressConfig()
    kwargs = {}
    for key, value in OmegaConf.to_container(progress_node, resolve=True).items():
        if key == "model_version_policy" and isinstance(value, dict):
            kwargs[key] = ModelVersionPolicy(mode=value.get("mode", "exact"))
        else:
            kwargs[key] = value
    return ProgressConfig(**kwargs)


@patch(FullyAsyncRollouter, "fit")
async def _rollouter_fit(self):
    """Start the async rollouter — FT-aware Supervisor lifecycle."""

    print("[FullyAsyncRollouter] Starting FullyAsyncRollouter...")

    if self.message_queue_client is None:
        raise ValueError("MessageQueue client not set. Call set_message_queue_client() first.")

    # The Supervisor normally started in init_ft_supervisor, before the
    # trainer's initial sync. Keep a guarded fallback for direct callers.
    if getattr(self, "_ft_supervisor", None) is not None:
        import logging as _ft_logging

        if not self._ft_supervisor.is_running:
            _ft_logging.getLogger(__name__).warning("[FT] fit: starting Supervisor heartbeat")
            self._ft_supervisor.start()
        else:
            _ft_logging.getLogger(__name__).debug("[FT] fit: Supervisor heartbeat already running")
    else:
        import logging as _ft_logging

        _ft_logging.getLogger(__name__).warning("[FT] fit: _ft_supervisor is None — no FT detection")

    # Set the running status flag
    async with self.lock:
        self.paused = False
        self.running = True
        self._resume_event.set()

    # Create the main asynchronous task
    generation_task = safe_create_task(self._streaming_generation_main(), name="generation_task")
    monitor_task = safe_create_task(self._async_monitor_loop(), name="monitor_task")

    try:
        # Run build and monitoring tasks concurrently
        await asyncio.gather(generation_task, monitor_task, return_exceptions=True)
    except Exception as e:
        print(f"[FullyAsyncRollouter] Asynchronous task execution error: {e}")
    finally:
        if not generation_task.done():
            generation_task.cancel()
        if not monitor_task.done():
            monitor_task.cancel()

        # Wait for the task to complete
        await asyncio.gather(generation_task, monitor_task, return_exceptions=True)

        # FT: stop Supervisor heartbeat
        if getattr(self, "_ft_supervisor", None) is not None:
            self._ft_supervisor.stop()

    print("[FullyAsyncRollouter] Rollouter fit completed")


# ---------------------------------------------------------------------------
# fully_async_policy.fully_async_trainer — FullyAsyncTrainer
# ---------------------------------------------------------------------------
@patch(FullyAsyncTrainer, "_setup_checkpoint_manager")
def _async_trainer_setup_checkpoint_manager(self, rollouter):
    """Setup checkpoint manager after rollouter is initialized (FT-aware)."""
    replicas = ray.get(rollouter.get_replicas.remote())
    checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)

    # FT: read fault_tolerance config + fetch LB handle from rollouter
    ft_cfg = None
    try:
        from verl.workers.rollout.fault_tolerance import FaultToleranceConfig

        ft_node = OmegaConf.select(self.config, "async_training.fault_tolerance")
        if ft_node is not None:
            ft_cfg = FaultToleranceConfig(**OmegaConf.to_container(ft_node, resolve=True))
    except Exception:
        ft_cfg = None

    lb_handle = None
    try:
        lb_handle = ray.get(rollouter.get_load_balancer.remote())
    except Exception:
        lb_handle = None

    self.checkpoint_manager = CheckpointEngineManager(
        config=checkpoint_engine_config,
        trainer=self.actor_wg,
        replicas=replicas,
        fault_tolerance=ft_cfg,
        load_balancer_handle=lb_handle,
        sync_failure_reporter=(lambda replica_id, source: rollouter.report_sync_failure.remote(replica_id, source)),
        replica_promotion_reporter=(
            lambda replica_id, servers, attempt_id, target_version: (
                rollouter.promote_synced_replica.remote(
                    replica_id,
                    servers,
                    attempt_id,
                    target_version,
                )
            )
        ),
    )
    print("[FullyAsyncTrainer] Checkpoint manager initialized")


@add(FullyAsyncTrainer, "_on_replica_dead_from_supervisor")
async def _async_trainer_on_replica_dead_from_supervisor(self, replica_id: str):
    """Cross-actor callback: Rollouter's Supervisor detected a dead replica.

    Prunes the replica from the trainer-side CKE and marks membership dirty
    so the next update_weights rebuilds the NCCL group without it.
    """
    if self.checkpoint_manager is not None:
        await self.checkpoint_manager.on_replica_dead(replica_id)


@add(FullyAsyncTrainer, "_on_replica_added_from_supervisor")
async def _async_trainer_on_replica_added_from_supervisor(self, new_replica):
    """Cross-actor callback: Rollouter's Supervisor spawned a replacement.

    Register the replacement as pending.  The next complete weight-sync
    transaction promotes it and only then admits it to the load balancer.
    """
    if self.checkpoint_manager is not None:
        self.checkpoint_manager.add_pending_replicas([new_replica])


# ---------------------------------------------------------------------------
# fully_async_policy.fully_async_main — FullyAsyncTaskRunner
# ---------------------------------------------------------------------------
@patch(FullyAsyncTaskRunner, "_initialize_components")
def _async_main_initialize_components(self, config) -> None:
    """Initialize all components for the fully-async training run (FT-aware)."""
    import os
    import socket
    from concurrent.futures import ThreadPoolExecutor
    from pprint import pprint

    import ray
    from omegaconf import OmegaConf

    from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient
    from verl.experimental.separation.utils import create_role_worker_mapping
    from verl.utils.fs import copy_to_local

    print(f"[ASYNC MAIN] TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    print("[ASYNC MAIN] Initializing model and tokenizer...")
    local_path = copy_to_local(
        config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
    )
    from verl.utils import hf_processor, hf_tokenizer

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    # Used for multimodal LLM, could be None
    processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

    self.components["tokenizer"] = tokenizer
    self.components["processor"] = processor
    self.components["config"] = config

    print("[ASYNC MAIN] Creating worker mapping and resource pools...")
    role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)
    self.components["role_worker_mapping"] = role_worker_mapping
    self.components["ray_worker_group_cls"] = ray_worker_group_cls

    print("[ASYNC MAIN] Creating FullyAsyncRollouter and FullyAsyncTrainer in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Rollouter does not permit continuous allocation, so we allocate trainer first.
        trainer_future = executor.submit(self._create_trainer, config)
        trainer_future.result()

        rollouter_future = executor.submit(self._create_rollouter, config)
        rollouter_future.result()

    # sync total_train_steps between rollouter and trainer
    total_train_steps = ray.get(self.components["rollouter"].get_total_train_steps.remote())
    print(f"total_train_steps {total_train_steps}")
    ray.get(self.components["trainer"].set_total_train_steps.remote(total_train_steps))

    # max_queue_size
    max_queue_size = ray.get(self.components["rollouter"].get_max_queue_size.remote())
    print(f"[ASYNC MAIN] Creating MessageQueue... max_queue_size {max_queue_size}")
    message_queue = MessageQueue.remote(config, max_queue_size)
    message_queue_client = MessageQueueClient(message_queue)
    self.components["message_queue"] = message_queue
    self.components["message_queue_client"] = message_queue_client

    ray.get(self.components["rollouter"].set_message_queue_client.remote(self.components["message_queue_client"]))
    ray.get(self.components["trainer"].set_message_queue_client.remote(self.components["message_queue_client"]))

    # param_version resume from ckpt or default 0
    ray.get(self.components["trainer"].load_checkpoint.remote())
    ray.get(self.components["rollouter"].load_checkpoint.remote())

    print("[ASYNC MAIN] Setting up parameter synchronization...")
    ray.get(self.components["trainer"].set_rollouter.remote(self.components["rollouter"]))

    # FT: wire Rollouter's Supervisor to Trainer's CKE (cross-actor callbacks).
    # Must run after set_rollouter so the trainer's checkpoint_manager exists.
    ray.get(self.components["rollouter"].init_ft_supervisor.remote(self.components["trainer"]))

    print("[ASYNC MAIN] Param sync before fit..")
    ray.get(self.components["trainer"]._fit_update_weights.remote())

    if config.trainer.get("val_before_train", True):
        ray.get(self.components["trainer"]._fit_validate.remote(True))

    print("[ASYNC MAIN] All components initialized successfully")
