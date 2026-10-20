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

import verl.trainer.main_ppo
from verl.checkpoint_engine import CheckpointEngineManager
from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorker,
    get_trajectory_info,
)
from verl.experimental.fully_async_policy import (
    fully_async_main,
    fully_async_rollouter,
    fully_async_trainer,
)
from verl.experimental.fully_async_policy.detach_utils import safe_create_task
from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncAgentLoopManager,
    FullyAsyncRollouter,
)
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from verl.experimental.reward_loop.reward_loop import RewardLoopManager
from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.protocol import DataProto
from verl.single_controller.ray.base import RayResourcePool
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.ray_utils import auto_await
from verl.utils.rollout_trace import RolloutTraceConfig
from verl.utils.tracking import Tracking
from verl.workers.rollout.fault_tolerance import filter_partial_batch
from verl.workers.rollout.llm_server import LLMServerManager

from ._core import add, patch, patch_module_function, unwrap_ray_remote, wrap

logger = logging.getLogger(__name__)


def _ft_build_fault_tolerance_config(config):
    """Build ``FaultToleranceConfig`` from ``async_training.fault_tolerance``.

    Returns ``None`` when the section is absent. Errors are NOT swallowed:
    a malformed section (unknown key, bad nested value) must fail the run
    loudly — silently treating it as disabled would drop Supervisor/CKE
    protection unnoticed.
    """
    from verl.workers.rollout.fault_tolerance import FaultToleranceConfig

    ft_node = OmegaConf.select(config, "async_training.fault_tolerance")
    if ft_node is None:
        return None
    return FaultToleranceConfig(**OmegaConf.to_container(ft_node, resolve=True))


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

    ft_cfg = _ft_build_fault_tolerance_config(self.config)

    from verl.utils.config import omega_conf_to_dataclass

    self.checkpoint_manager = CheckpointEngineManager(
        config=omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine),
        trainer=self.actor_rollout_wg,
        # Copy: get_replicas() returns the manager's live rollout_replicas list;
        # CKE would prune it in place on replica death, breaking spawn_replacement.
        replicas=list(self.llm_server_manager.get_replicas()),
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

        # one-step control-plane faults: LLMServerManager.clear_kv_cache hands
        # transient rollout faults here instead of failing the weight-sync step
        # (the Manager keeps dead replicas listed for spawn_replacement). Fixed
        # order — the Supervisor must be running (report_failure silently no-ops
        # otherwise), LB isolation first so new acquires stop routing to the
        # dead replica, then the idempotent report that drives CKE pruning and
        # replacement through the existing dead-queue flow.
        if getattr(self, "llm_server_manager", None) is not None:

            async def _ft_clear_kv_fault_callback(replica_id: str) -> None:
                sup = self._ft_supervisor
                if sup is None or not sup.is_running:
                    raise RuntimeError(f"[FT] clear_kv_cache fault on replica {replica_id}: Supervisor is not running")
                await asyncio.wait_for(
                    self.llm_server_manager.global_load_balancer.remove_servers.remote([replica_id]),
                    timeout=10.0,
                )
                sup.report_failure(replica_id, "clear_kv_cache")

            self.llm_server_manager._ft_clear_kv_fault_callback = _ft_clear_kv_fault_callback

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

    dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")
    rem = len(batch) % dp_size
    if rem and len(batch) > dp_size:
        batch = batch[: len(batch) - rem]

    if "response_mask" not in batch.batch.keys():
        batch.batch["response_mask"] = compute_response_mask(batch)
    # Balance the number of valid tokens across DP ranks.
    # NOTE: This usually changes the order of data in the `batch`,
    # which won't affect the advantage calculation (since it's based on uid),
    # but might affect the loss calculation (due to the change of mini-batching).
    if self.config.trainer.balance_batch and len(batch) > 0 and len(batch) % dp_size == 0:
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
    if not OmegaConf.select(self.config, "async_training.fault_tolerance.enabled", default=False):
        return
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

    ft_cfg = _ft_build_fault_tolerance_config(self.config)

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
    if not OmegaConf.select(self.config, "async_training.fault_tolerance.enabled", default=False):
        return await self._orig__init_async_rollout_manager()

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
    if not OmegaConf.select(self.config, "async_training.fault_tolerance.enabled", default=False):
        return await self._orig_fit()

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
    if not OmegaConf.select(self.config, "async_training.fault_tolerance.enabled", default=False):
        return self._orig__setup_checkpoint_manager(rollouter)

    replicas = ray.get(rollouter.get_replicas.remote())
    checkpoint_engine_config = omega_conf_to_dataclass(self.config.actor_rollout_ref.rollout.checkpoint_engine)

    # FT: read fault_tolerance config + fetch LB handle from rollouter
    ft_cfg = _ft_build_fault_tolerance_config(self.config)

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


@patch(FullyAsyncTrainer, "_get_samples_from_queue")
async def _async_trainer_get_samples_from_queue(self):
    """FT guard: drop partial rollout samples (rows < rollout.n) instead of
    assembling them, which would break dp divisibility in ``_balance_batch``
    and bias GRPO uid groups. Keep waiting for complete ones; ``None`` (the
    termination signal) still ends collection.
    """
    if not OmegaConf.select(self.config, "async_training.fault_tolerance.enabled", default=False):
        return await self._orig__get_samples_from_queue()

    import logging
    import time

    import ray

    from verl.experimental.fully_async_policy.detach_utils import assemble_batch_from_rollout_samples

    log = logging.getLogger(__name__)
    expected_rows = self.config.actor_rollout_ref.rollout.n

    print(
        f"[FullyAsyncTrainer] Requesting {self.required_samples} samples from queue",
        flush=True,
    )

    consumer_start = time.time()
    queue_samples = []
    queue_len = 0
    dropped = 0
    while len(queue_samples) < self.required_samples:
        sample, queue_len = await self.message_queue_client.get_sample()

        if sample is None:
            print(
                f"[FullyAsyncTrainer] Detected termination signal (None), stopping sample collection. "
                f"Collected {len(queue_samples)}/{self.required_samples} samples"
            )
            break

        rollout_sample = ray.cloudpickle.loads(sample)
        if len(rollout_sample.full_batch) != expected_rows:
            dropped += 1
            log.warning(
                "[FT] trainer dropped partial rollout sample %s: %d/%d rows (dropped_total=%d)",
                getattr(rollout_sample, "sample_id", "?"),
                len(rollout_sample.full_batch),
                expected_rows,
                dropped,
            )
            continue

        queue_samples.append(rollout_sample)

        if len(queue_samples) % 64 == 0:
            print(
                f"[FullyAsyncTrainer] Collected {len(queue_samples)}/{self.required_samples} samples. "
                f"mq_len: {queue_len}"
            )

    consumer_end = time.time()

    if not queue_samples or len(queue_samples) < self.required_samples:
        print("[FullyAsyncTrainer] not enough samples collected after loop")
        return None, None
    total_wait_time = consumer_end - consumer_start

    print(
        f"[FullyAsyncTrainer] Loop collection completed: {len(queue_samples)}/{self.required_samples} samples, "
        f"total wait time: {total_wait_time:.2f} seconds. "
        f"mq_len: {queue_len}"
    )

    if dropped:
        log.warning("[FT] trainer skipped %d partial rollout sample(s) while collecting this batch", dropped)

    # Assemble batch - now working directly with RolloutSample objects
    if self.config.trainer.balance_batch:
        batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, self._balance_batch)
    else:
        batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config, None)

    batch.meta_info["fully_async/total_wait_time"] = total_wait_time
    return 0, batch


# ---------------------------------------------------------------------------
# fully_async_policy.fully_async_main — FullyAsyncTaskRunner
# ---------------------------------------------------------------------------
@patch(FullyAsyncTaskRunner, "_initialize_components")
def _async_main_initialize_components(self, config) -> None:
    """Initialize all components for the fully-async training run (FT-aware)."""
    if not OmegaConf.select(config, "async_training.fault_tolerance.enabled", default=False):
        return self._orig__initialize_components(config)

    import os
    import socket
    from concurrent.futures import ThreadPoolExecutor
    from pprint import pprint

    import ray

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
    message_queue = _FtNonInferenceRemote(MessageQueue, "message_queue").remote(config, max_queue_size)
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


# ---------------------------------------------------------------------------
# placement — keep CPU-side actors and standalone rollout replicas apart
# ---------------------------------------------------------------------------
# Deployment contract: nodes carry a role label set at ray start time
# (`ray start --labels='{"verl.io/role": "<role>"}'`): trainer nodes run the
# training + CPU coordination actors, rollout nodes run standalone rollout
# replicas, and the head runs no business workload (it carries the head label
# only, so both hard constraints below exclude it).
# - CPU-side coordination actors use a strict trainer-label constraint.
# - Standalone rollout placement groups (CKE workers + the vLLM HTTP servers
#   that follow them) use a strict rollout-label constraint; native
#   scheduling would otherwise let them land on trainer nodes.
# - Every other placement group (training GPU workers, CPU coordination
#   pools) uses a strict trainer-label constraint; otherwise a training
#   bundle could occupy a rollout pod, which must only ever run
#   inference-instance processes, and the rollout PG would stay pending.
# Requires Ray >= 2.49 for placement-group bundle_label_selector; enforced
# by _ft_check_ray_label_support at startup.
# Hybrid (colocated) rollout keeps native scheduling: it shares the trainer
# pool by design.


def _ft_placement_enabled(config) -> bool:
    """Placement is opt-out: on whenever fault tolerance is enabled."""
    if config is None:
        return False
    if not OmegaConf.select(config, "async_training.fault_tolerance.enabled", default=False):
        return False
    return OmegaConf.select(config, "async_training.fault_tolerance.placement.enabled", default=True)


def _ft_check_ray_label_support() -> None:
    """Fail fast when the running Ray cannot schedule by node labels.

    Placement groups accept ``bundle_label_selector`` only from Ray 2.49
    (see the Ray node-labels docs); verl's floor (>= 2.41) is NOT enough.
    Runs once per process, at first activation of placement.
    """
    if getattr(_ft_check_ray_label_support, "_checked", False):
        return
    import re

    version = getattr(ray, "__version__", "") or ""
    match = re.match(r"(\d+)\.(\d+)", version)
    if match is None:
        logger.warning(
            "[placement] cannot parse Ray version %r; skipping the node-label support check "
            "(placement requires Ray >= 2.49)",
            version,
        )
    elif (int(match.group(1)), int(match.group(2))) < (2, 49):
        raise RuntimeError(
            f"[placement] Ray {version} does not support placement-group bundle_label_selector "
            "(requires Ray >= 2.49; the pinned test environment is Ray 2.55.1). Upgrade Ray or set "
            "async_training.fault_tolerance.placement.enabled=False."
        )
    _ft_check_ray_label_support._checked = True


_ft_label_cfg = None


def _ft_cache_label_config(config) -> None:
    """Cache the role-label config for call sites that have no config (PGs)."""
    global _ft_label_cfg
    if config is None or not _ft_placement_enabled(config):
        return
    _ft_check_ray_label_support()
    _ft_label_cfg = {
        "key": OmegaConf.select(
            config, "async_training.fault_tolerance.placement.node_label_key", default="verl.io/role"
        )
        or "",
        "trainer": OmegaConf.select(
            config, "async_training.fault_tolerance.placement.trainer_label_value", default="trainer"
        )
        or "",
        "rollout": OmegaConf.select(
            config, "async_training.fault_tolerance.placement.rollout_label_value", default="rollout"
        )
        or "",
    }


def _ft_non_inference_scheduling_strategy(config, label: str):
    """Strict node-label strategy constraining an actor to trainer nodes.

    Returns ``None`` when placement is disabled, the label key/value is empty
    (disables the constraint), or the Ray runtime cannot express node-label
    constraints (missing from very old Ray; placement as a whole requires
    >= 2.49 anyway, enforced by ``_ft_check_ray_label_support``).
    """
    _ft_cache_label_config(config)
    if not _ft_placement_enabled(config):
        return None
    key = OmegaConf.select(
        config,
        "async_training.fault_tolerance.placement.node_label_key",
        default="verl.io/role",
    )
    value = OmegaConf.select(
        config,
        "async_training.fault_tolerance.placement.trainer_label_value",
        default="trainer",
    )
    if not key or not value:
        return None
    try:
        from ray.util.scheduling_strategies import In, NodeLabelSchedulingStrategy
    except Exception:  # pragma: no cover - defensive: Ray without node labels
        return None
    # A hard label constraint silently parks the actor when no node matches;
    # warn once instead of leaving an unexplained PENDING actor.
    if not getattr(_ft_non_inference_scheduling_strategy, "_label_warned", False):
        _ft_non_inference_scheduling_strategy._label_warned = True
        try:
            labelled = any(node.get("Alive") and (node.get("Labels") or {}).get(key) == value for node in ray.nodes())
            if not labelled:
                logger.warning(
                    "[placement] no alive node carries label %s=%s yet; %s will stay pending "
                    "until a trainer node joins with 'ray start --labels'",
                    key,
                    value,
                    label,
                )
        except Exception:  # pragma: no cover - defensive: no cluster metadata yet
            pass
    logger.info("[placement] schedule %s on nodes labelled %s=%s (strict)", label, key, value)
    return NodeLabelSchedulingStrategy(hard={key: In(value)})


def _ft_rollout_pg_label_selector(name_prefix: str):
    """Per-bundle label selector pinning a placement group to rollout nodes.

    Applies only to standalone-rollout pools (names starting with
    ``rollout``). Returns ``None`` for other pools (they are pinned to
    trainer nodes by ``_ft_trainer_pg_label_selector``), when no label
    config is cached, or when the rollout label value is empty (disables
    the constraint).
    """
    if not name_prefix.startswith("rollout"):
        return None
    cfg = globals().get("_ft_label_cfg")  # globals().get: also safe under partial-module exec (tests)
    if not cfg:
        # run_ppo never ran with a config (custom entry point): stay native.
        if not getattr(_ft_rollout_pg_label_selector, "_no_cfg_warned", False):
            _ft_rollout_pg_label_selector._no_cfg_warned = True
            logger.warning("[placement] no label config cached; rollout placement groups use native scheduling")
        return None
    key, value = cfg["key"], cfg["rollout"]
    if not key or not value:
        return None
    logger.info("[placement] schedule rollout pool %s on nodes labelled %s=%s (strict)", name_prefix, key, value)
    return {key: value}


def _ft_trainer_pg_label_selector(name_prefix: str):
    """Per-bundle label selector pinning a (non-rollout) pool to trainer nodes.

    Training GPU workers share ``trainer_pool``; reward/teacher pools and any
    other non-rollout pool stay on trainer nodes too, so no training bundle
    can ever occupy a rollout pod. Returns ``None`` for standalone-rollout
    pools (handled by ``_ft_rollout_pg_label_selector``), when no label config
    is cached, or when the trainer label value is empty.
    """
    if name_prefix.startswith("rollout"):
        return None
    cfg = globals().get("_ft_label_cfg")  # globals().get: also safe under partial-module exec (tests)
    if not cfg:
        # run_ppo never ran with a config (custom entry point): stay native.
        if not getattr(_ft_trainer_pg_label_selector, "_no_cfg_warned", False):
            _ft_trainer_pg_label_selector._no_cfg_warned = True
            logger.warning("[placement] no label config cached; pool %s uses native scheduling", name_prefix)
        return None
    key, value = cfg["key"], cfg["trainer"]
    if not key or not value:
        return None
    logger.info("[placement] schedule pool %s on nodes labelled %s=%s (strict)", name_prefix, key, value)
    return {key: value}


def _ft_require_labelled_node(key: str, value: str, what: str):
    """Check that at least one alive node carries the strict role label.

    Returns ``(ok, cluster_detail)``; callers turn a miss into a loud,
    self-explanatory failure instead of a silently pending placement group.
    """
    try:
        nodes = ray.nodes()
    except Exception:  # pragma: no cover - defensive: no cluster metadata yet
        nodes = []
    for node in nodes:
        if node.get("Alive") and (node.get("Labels") or {}).get(key) == value:
            return True, ""
    detail = (
        "; ".join(f"{node.get('NodeManagerAddress', '?')}: {(node.get('Labels') or {}).get(key)!r}" for node in nodes)
        or "no nodes registered"
    )
    return False, detail


def _ft_labelled_get_placement_groups(self, selector, what, strategy="STRICT_PACK", name=None, device_name="cuda"):
    """Native ``RayResourcePool.get_placement_groups`` plus a label selector.

    Mirrors verl's implementation verbatim; the only addition is
    ``bundle_label_selector`` so every bundle lands on a node labelled for
    the pool's role (``what``: "rollout" or "trainer").
    ``bundle_label_selector`` requires Ray >= 2.49 — enforced by
    ``_ft_check_ray_label_support`` at startup.
    """
    from ray.util.placement_group import placement_group

    from verl.single_controller.ray.base import sort_placement_group_by_node_ip

    if self.pgs is not None:
        return self.pgs
    pg_name_prefix = (
        name if name else f"{self.name_prefix}verl_group_{'_'.join([str(count) for count in self._store])}:"
    )
    key, value = next(iter(selector.items()))
    ok, detail = _ft_require_labelled_node(key, value, what)
    if not ok:
        # Strict whitelist isolation: no native-scheduling fallback, so a
        # mislabelled cluster must fail loudly rather than silently place
        # the workload on the wrong pool (or hang forever pending).
        raise RuntimeError(
            f"[placement] no alive node carries label {key}={value} for the {what} placement group "
            f"'{pg_name_prefix}' (alive nodes: {detail}). Add the node with "
            f'\'ray start --labels={{"{key}": "{value}"}}\' to unblock.'
        )
    if device_name == "npu":
        device_name = "NPU"
    elif device_name == "cuda":
        device_name = "GPU"
    bundle = {"CPU": self.max_colocate_count}
    if self.use_gpu:
        bundle[device_name] = 1
        if self.accelerator_type is not None:
            bundle[self.accelerator_type] = 1e-4
    pg_scheme = [[bundle.copy() for _ in range(process_count)] for process_count in self._store]
    lifetime = "detached" if self.detached else None
    pgs = [
        placement_group(
            bundles=bundles,
            strategy=strategy,
            name=pg_name_prefix + str(idx),
            lifetime=lifetime,
            bundle_label_selector=[dict(selector) for _ in bundles],
        )
        for idx, bundles in enumerate(pg_scheme)
    ]
    ray.get([pg.ready() for pg in pgs])
    self.pgs = sort_placement_group_by_node_ip(pgs)
    return pgs


@wrap(RayResourcePool, "get_placement_groups")
def _ft_resource_pool_get_placement_groups(orig, self, *args, **kwargs):
    """Create placement groups on role-labelled nodes.

    Standalone-rollout pools (``rollout*``) land on rollout nodes; every
    other pool — training GPU workers and the CPU coordination pools — lands
    on trainer nodes, so a training bundle can never occupy a rollout pod.
    """
    selector = _ft_rollout_pg_label_selector(self.name_prefix)
    what = "rollout"
    if selector is None:
        selector = _ft_trainer_pg_label_selector(self.name_prefix)
        what = "trainer"
    if selector is None:
        return orig(self, *args, **kwargs)
    return _ft_labelled_get_placement_groups(self, selector, what, *args, **kwargs)


@wrap(SeparateRayPPOTrainer, "_init_resource_pools")
def _ft_cache_labels_before_resource_pools(orig, self, *args, **kwargs):
    """Cache the role-label config right before resource pools create PGs.

    The process that creates placement groups (the trainer actor) may never
    have gone through ``run_ppo``/a reward manager; without this, its PGs
    would silently fall back to native scheduling. Also covers the
    fully-async trainer, which inherits ``_init_resource_pools``.
    """
    _ft_cache_label_config(self.config)
    return orig(self, *args, **kwargs)


class _FtNonInferenceRemote:
    """Proxy exposing ``.remote()`` that keeps an actor off inference nodes.

    Applies ``_ft_non_inference_scheduling_strategy`` when the caller passes a
    config (positionally or as the ``config`` kwarg); otherwise falls through
    to native scheduling.
    """

    def __init__(self, actor_cls, label: str):
        self._actor_cls = actor_cls
        self._label = label

    def remote(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        strategy = _ft_non_inference_scheduling_strategy(config, self._label)
        if strategy is None:
            return self._actor_cls.remote(*args, **kwargs)
        return self._actor_cls.options(scheduling_strategy=strategy).remote(*args, **kwargs)


@wrap(AgentLoopManager, "_init_agent_loop_workers")
async def _agent_manager_init_workers(orig, self, *args, **kwargs):
    """Create agent loop workers under the non-inference label constraint."""
    strategy = _ft_non_inference_scheduling_strategy(self.config, "agent loop workers")
    if strategy is None:
        return await orig(self, *args, **kwargs)

    self.agent_loop_workers = []
    num_workers = self.rollout_config.agent.num_workers
    for i in range(num_workers):
        self.agent_loop_workers.append(
            self.agent_loop_workers_class.options(
                name=f"agent_loop_worker_{i}" + f"_{uuid.uuid4().hex[:8]}",
                scheduling_strategy=strategy,
            ).remote(
                self.config,
                self.llm_client,
                self.teacher_client,
                self.reward_loop_worker_handles,
            )
        )
    logger.info("[placement] placed %d agent loop workers under the non-inference label constraint", num_workers)


@wrap(RewardLoopManager, "_init_reward_loop_workers")
def _reward_manager_init_workers(orig, self, *args, **kwargs):
    """Create reward loop workers under the non-inference label constraint."""
    strategy = _ft_non_inference_scheduling_strategy(self.config, "reward loop workers")
    if strategy is None:
        return orig(self, *args, **kwargs)

    self.reward_loop_workers = []
    num_workers = self.config.reward.num_workers
    for i in range(num_workers):
        self.reward_loop_workers.append(
            self.reward_loop_workers_class.options(
                name=f"reward_loop_worker_{i}",
                scheduling_strategy=strategy,
            ).remote(self.config, self.reward_router_address)
        )
    logger.info("[placement] placed %d reward loop workers under the non-inference label constraint", num_workers)


# ---------------------------------------------------------------------------
# one-step-off / legacy / fully-async entries: keep the TaskRunner off
# inference nodes
# ---------------------------------------------------------------------------
# Native scheduling would let the runner actor float onto a CPU-rich
# inference pod that Kubernetes deletes on fault, so wrap the runner class
# with the strict label constraint. It must stay a real Ray actor: running it
# in the driver would import the NPU-dependent worker classes in the
# (NPU-less) driver process.


class _FtTaskRunnerRemote:
    """Proxy exposing ``.remote()`` / ``.options()`` that keeps the task runner
    off inference nodes (see ``_ft_non_inference_scheduling_strategy``).

    The constraint is strict, so the actor stays pending instead of floating
    onto an inference pod. ``run_ppo`` only touches ``.remote()`` (plus
    ``.options(...)`` for the nsys runtime env) before blocking on
    ``ray.get(runner.run.remote(config))``.
    """

    def __init__(self, actor_cls, config, label: str):
        self._actor_cls = actor_cls
        self._config = config
        self._label = label

    def options(self, **options):
        strategy = _ft_non_inference_scheduling_strategy(self._config, self._label)
        if strategy is not None:
            options["scheduling_strategy"] = strategy
        return self._actor_cls.options(**options)

    def remote(self, *args, **kwargs):
        return self.options().remote(*args, **kwargs)


@patch_module_function(verl.trainer.main_ppo, "run_ppo")
def _run_ppo(config, task_runner_class=None):
    _ft_cache_label_config(config)
    if task_runner_class is None:
        # Mirror native main()'s default: the runner class is already a Ray
        # actor class on the pinned verl baseline (there is no V1 class).
        task_runner_class = getattr(verl.trainer.main_ppo, "TaskRunner", None)
    if (
        task_runner_class is not None
        and not isinstance(task_runner_class, (_FtNonInferenceRemote, _FtTaskRunnerRemote))
        and _ft_placement_enabled(config)
    ):
        logger.info("[placement] FT placement: pin the task runner away from inference nodes")
        task_runner_class = _FtTaskRunnerRemote(task_runner_class, config, "the task runner")
    return verl.trainer.main_ppo._orig_run_ppo(config, task_runner_class)


# Refresh Ray's wrappers and method tables only after all decorators complete.
# Reuse the original Python classes to preserve identity, inheritance and super().
FullyAsyncRollouter = ray.remote(num_cpus=10, max_concurrency=100)(unwrap_ray_remote(FullyAsyncRollouter))
FullyAsyncTrainer = ray.remote(num_cpus=10)(unwrap_ray_remote(FullyAsyncTrainer))
fully_async_rollouter.FullyAsyncRollouter = FullyAsyncRollouter
fully_async_trainer.FullyAsyncTrainer = FullyAsyncTrainer
fully_async_main.FullyAsyncRollouter = FullyAsyncRollouter
fully_async_main.FullyAsyncTrainer = FullyAsyncTrainer

FullyAsyncTaskRunner = ray.remote(num_cpus=1)(unwrap_ray_remote(FullyAsyncTaskRunner))
fully_async_main.FullyAsyncTaskRunner = FullyAsyncTaskRunner

# Placement: keep the top-level CPU coordinators off inference nodes (see
# _FtNonInferenceRemote).
fully_async_main.FullyAsyncRollouter = _FtNonInferenceRemote(FullyAsyncRollouter, "fully_async_rollouter")
fully_async_main.FullyAsyncTrainer = _FtNonInferenceRemote(FullyAsyncTrainer, "fully_async_trainer")
