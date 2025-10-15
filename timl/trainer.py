# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py
# and https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py

import logging
import math
import os
import time
from collections.abc import Callable, Generator
from contextlib import nullcontext
from typing import Any

import psutil
import torch
import torch.distributed as dist
import wandb
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import enable_full_determinism, set_seed
from transformers.training_args import OptimizerNames
from transformers.utils.import_utils import (
    _is_package_available,
    is_liger_kernel_available,
)

from timl.data_loader_tokens import get_num_tokens
from timl.distributed import get_rank, init_distributed, is_dist_available, print0
from timl.trainer_metric import TrainerMetric
from timl.training_args import TrainingArguments
from timl.upload_checkpoint_mixin import UploadCheckpointMixin

logger = logging.getLogger(__name__)


DataLoaderFn = Callable[[str, int, int], Generator[dict[str, Tensor], None, None]]
ComputeLossFunc = Callable[[CausalLMOutputWithPast, dict[str, Tensor]], Tensor]


class Trainer(UploadCheckpointMixin[TrainingArguments]):
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        train_data_loader_fn: DataLoaderFn,
        eval_data_loader_fn: DataLoaderFn,
        optimizers: tuple[Optimizer, LRScheduler] | tuple[None, None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[Optimizer], dict[str, Any]] | None = None,
        compute_loss_func: ComputeLossFunc | None = None,
        train_metric_cls: type[TrainerMetric] | list[type[TrainerMetric]] | None = None,
        eval_metric_cls: type[TrainerMetric] | list[type[TrainerMetric]] | None = None,
        config: dict | None = None,  # Extra config for wandb
    ):
        self.args = args

        self.is_dist_available = is_dist_available()
        self.rank, self.local_rank, self.world_size, self.device, self.main_process = (
            init_distributed()
        )

        enable_full_determinism(args.seed) if args.full_determinism else set_seed(
            args.seed
        )

        cpu_count = psutil.cpu_count(logical=False)
        if (
            os.environ.get("OMP_NUM_THREADS") == "1" and cpu_count is not None
        ):  # Physical cores / processes
            omp_num_threads = max(cpu_count // self.world_size, 1)
            print0(
                f"Setting OMP_NUM_THREADS environment variable for each process to be {omp_num_threads}"
            )
            os.environ["OMP_NUM_THREADS"] = str(omp_num_threads)

        if self.args.use_liger_kernel:
            if is_liger_kernel_available():
                from liger_kernel.transformers import (  # ty: ignore[unresolved-import]
                    _apply_liger_kernel_to_instance,
                )

                liger_kernel_kwargs = args.liger_kernel_kwargs or {}
                _apply_liger_kernel_to_instance(model=model, **liger_kernel_kwargs)
            else:
                logger.warning("`liger_kernel` package is not available.")

        if self.args.use_cut_cross_entropy:
            if _is_package_available("cut_cross_entropy"):
                from cut_cross_entropy.transformers import (  # ty: ignore[unresolved-import]
                    cce_patch,
                )

                cut_cross_entropy_kwargs = args.cut_cross_entropy_kwargs or {}
                try:
                    model = cce_patch(model, **cut_cross_entropy_kwargs)
                except RuntimeError as error:
                    logger.warning("`cce_patch` failed: %s", error)
            else:
                logger.warning("`cut_cross_entropy` package is not available.")

        self.train_dataloader_fn = train_data_loader_fn
        self.eval_dataloader_fn = eval_data_loader_fn

        model = model.to(self.device)
        self.model = self.ddp_model = model
        if self.is_dist_available:
            self.ddp_model = DistributedDataParallel(
                model, device_ids=[self.local_rank]
            )
            self.model = self.ddp_model.module

        self.compute_loss_func = compute_loss_func
        self.train_metric_cls = (
            [train_metric_cls]
            if isinstance(train_metric_cls, type)
            else train_metric_cls
        )
        self.eval_metric_cls = (
            [eval_metric_cls] if isinstance(eval_metric_cls, type) else eval_metric_cls
        )

        self.optimizer, self.lr_scheduler = optimizers
        self.optimizer_cls_and_kwargs = optimizer_cls_and_kwargs
        if self.optimizer_cls_and_kwargs is not None and self.optimizer is not None:
            raise RuntimeError(
                "Passing both `optimizers` and `optimizer_cls_and_kwargs` is not allowed."
            )

        self.create_upload_queue()
        self.config = config or {}

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.create_optimizer()
        self.create_scheduler(num_training_steps, self.optimizer)

    def get_decay_parameter_names(self, model: PreTrainedModel) -> list[str]:
        forbidden_layer_names = [
            r"bias",
            r"layernorm",
            r"rmsnorm",
            r"(?:^|\.)norm(?:$|\.)",
            r"_norm(?:$|\.)",
        ]
        return get_parameter_names(
            model, [torch.nn.LayerNorm], forbidden_layer_names=forbidden_layer_names
        )

    def create_optimizer(self) -> Optimizer:
        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(self.model)
            optimizer_grouped_parameters = [
                self.get_grouped_parameters(
                    lambda n, p: n in decay_parameters,
                    weight_decay=self.args.weight_decay,
                ),
                self.get_grouped_parameters(
                    lambda n, p: n not in decay_parameters,
                    weight_decay=0.0,
                ),
            ]
            if self.optimizer_cls_and_kwargs is not None:
                optimizer_cls, optimizer_kwargs = self.optimizer_cls_and_kwargs
            else:
                optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                    self.args
                )
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )
        return self.optimizer

    def get_grouped_parameters(
        self, cond: Callable[[str, torch.nn.Parameter], bool], **kwargs
    ) -> dict[str, list[torch.nn.Parameter] | Any]:
        return {
            "params": [
                p
                for n, p in self.model.named_parameters()
                if cond(n, p) and p.requires_grad
            ],
            **kwargs,
        }

    @staticmethod
    def get_optimizer_cls_and_kwargs(
        args: TrainingArguments,
    ) -> tuple[type[Optimizer], dict[str, Any]]:
        optimizer_kwargs: dict = {"lr": args.learning_rate}

        adam_kwargs = {
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_epsilon,
        }

        if args.optim in [OptimizerNames.ADAMW_TORCH, OptimizerNames.ADAMW_TORCH_FUSED]:
            from torch.optim import AdamW

            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
            if args.optim == OptimizerNames.ADAMW_TORCH_FUSED:
                optimizer_kwargs.update({"fused": True})
        else:
            raise ValueError(f"Unsupported optimizer: {args.optim}")
        return optimizer_cls, optimizer_kwargs

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer | None = None
    ):
        if self.lr_scheduler is None:
            self.args.warmup_steps = self.args.warmup_steps or math.ceil(
                num_training_steps * self.args.warmup_ratio
            )
            optimizer = optimizer or self.optimizer
            if optimizer is None:
                raise RuntimeError("No optimizer found, cannot create LR scheduler.")
            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=num_training_steps,
                scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
            )
        return self.lr_scheduler

    def train(self) -> None:
        self.args.max_train_tokens = self.args.max_train_tokens or get_num_tokens(
            self.args.train_file_pattern
        )
        self.args.max_train_steps = self.args.max_train_steps or math.ceil(
            self.args.max_train_tokens
            / (self.args.train_batch_size * self.args.block_size)
        )

        self.args.max_eval_tokens = self.args.max_eval_tokens or get_num_tokens(
            self.args.eval_file_pattern
        )
        self.args.max_eval_steps = self.args.max_eval_steps or math.ceil(
            self.args.max_eval_tokens
            / (self.args.per_device_eval_batch_size * self.args.block_size)
        )

        self.args.gradient_accumulation_steps = self.args.train_batch_size // (
            self.args.per_device_train_batch_size * self.world_size
        )

        self.lr_scheduler = None
        self.create_optimizer_and_scheduler(self.args.max_train_steps)
        if self.optimizer is None:
            raise RuntimeError("No optimizer found, cannot train.")
        if self.lr_scheduler is None:
            raise RuntimeError("No LR scheduler found, cannot train.")

        object_list = [None]
        if self.main_process:  # Once we've set defaults
            import subprocess

            import yaml

            print(f"torch.version.__version__ {torch.version.__version__}")  # ty: ignore[unresolved-attribute]
            print(f"torch.version.cuda {torch.version.cuda}\n")  # ty: ignore[unresolved-attribute]
            print(
                f"{subprocess.run(['nvidia-smi'], capture_output=True, text=True).stdout}"
            )

            config = self.config
            config.update(self.args.__dict__)
            print(yaml.dump(config, explicit_start=True))

            params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            params_mib = (
                sum(
                    p.numel() * p.element_size()
                    for p in self.model.parameters()
                    if p.requires_grad
                )
                // 1024
                // 1024
            )
            print(f"params:{params} ({params_mib}MiB)")

            if not self.args.debug:
                project = self.args.wandb_project or self.args.output_dir
                run = wandb.init(
                    project=project, name=self.args.wandb_run_name, config=config
                )
                object_list = [run.id]
                print()

        if self.args.debug:
            self.args.max_train_steps = 1
            self.args.max_eval_steps = 1
            self.args.log_every_steps = 1
            self.args.save_every_steps = 1
            self.args.eval_every_steps = 1
            self.args.push_to_hub = False

        if not self.args.debug:
            if self.is_dist_available:  # Assumed rank 0 is the main process
                dist.broadcast_object_list(object_list, src=0)
                dist.barrier(device_ids=[self.local_rank])
            self.args.wandb_run_name = object_list[0]

        autocast = torch.autocast("cuda", torch.bfloat16, enabled=self.args.bf16)

        train_dataloader = self.train_dataloader_fn(
            self.args.train_file_pattern,
            self.args.per_device_train_batch_size,
            self.args.block_size,
        )
        train_time_ms, eval_time_ms, save_time_ms = 0, 0, 0
        eval_steps, save_steps = 0, 0

        for step in range(self.args.max_train_steps + 1):
            torch.cuda.synchronize()
            step_time_start = time.perf_counter()
            eval_step_time_ms = 0
            save_step_time_ms = 0

            is_last_step = step == self.args.max_train_steps
            is_eval_step = (
                self.args.eval_every_steps > 0
                and step % self.args.eval_every_steps == 0
            )
            is_save_step = (
                self.args.save_every_steps > 0
                and step % self.args.save_every_steps == 0
            )
            is_log_step = (
                self.args.log_every_steps > 0 and step % self.args.log_every_steps == 0
            )

            if is_last_step or is_eval_step:
                eval_step_time_start = time.perf_counter()

                self.ddp_model.eval()
                eval_dataloader = self.eval_dataloader_fn(
                    self.args.eval_file_pattern,
                    self.args.per_device_eval_batch_size,
                    self.args.block_size,
                )
                eval_loss = torch.zeros(1, device=self.device)
                eval_metrics = []
                if self.eval_metric_cls is not None:
                    eval_metrics = [
                        eval_metric_cls(self.args)
                        for eval_metric_cls in self.eval_metric_cls
                    ]

                num_eval_steps = self.args.max_eval_steps
                with torch.no_grad(), autocast:
                    for eval_step in range(self.args.max_eval_steps):
                        try:
                            inputs = next(eval_dataloader)
                            outputs: CausalLMOutputWithPast = self.ddp_model(
                                input_ids=inputs["input_ids"],
                                labels=inputs["labels"],
                                **(self.args.forward_kwargs or {}),
                            )
                            eval_loss += self.compute_loss(outputs, inputs)
                            for eval_metric in eval_metrics:
                                eval_metric.update(outputs, inputs)
                        except StopIteration:
                            num_eval_steps = eval_step + 1
                            break

                eval_loss /= num_eval_steps
                if self.is_dist_available:
                    dist.all_reduce(eval_loss, op=dist.ReduceOp.AVG)
                for eval_metric in eval_metrics:
                    eval_metric.gather()

                torch.cuda.synchronize()
                eval_step_time_ms = 1000 * (time.perf_counter() - eval_step_time_start)
                eval_time_ms += eval_step_time_ms

                if self.main_process:
                    eval_steps += 1
                    eval_step_avg = eval_time_ms / eval_steps
                    print(
                        f"step:{step}/{self.args.max_train_steps} "
                        f"eval_loss:{eval_loss.item():.4f} "
                        f"eval_time:{eval_time_ms:.0f}ms "
                        f"eval_step_avg:{eval_step_avg:.2f}ms"
                    )
                    eval_metric_data = {
                        "eval_loss": eval_loss.item(),
                        "eval_time": eval_time_ms,
                        "eval_step_avg": eval_step_avg,
                    }
                    for eval_metric in eval_metrics:
                        eval_metric_data.update(eval_metric.to_dict())
                    if not self.args.debug:
                        wandb.log(eval_metric_data, step=step)

                self.ddp_model.train()

            if is_last_step or is_save_step:
                save_step_time_start = time.perf_counter()

                self.save_checkpoint(step)
                if self.is_dist_available:
                    dist.barrier(device_ids=[self.local_rank])

                torch.cuda.synchronize()
                save_step_time_ms = 1000 * (time.perf_counter() - save_step_time_start)
                save_time_ms += save_step_time_ms

                if self.main_process:
                    save_steps += 1
                    save_step_avg = save_time_ms / save_steps
                    print(
                        f"step:{step}/{self.args.max_train_steps} "
                        f"save_time:{save_time_ms:.0f}ms "
                        f"save_step_avg:{save_step_avg:.2f}ms"
                    )

            if is_last_step:
                break

            train_loss, train_metrics = self.train_step(
                train_dataloader, autocast, is_log_step
            )

            if self.args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.ddp_model.parameters(), self.args.max_grad_norm
                )

            self.optimizer.step()
            self.lr_scheduler.step()
            self.ddp_model.zero_grad(set_to_none=True)

            # Loss is already divided by gradient accumulation steps
            if self.is_dist_available:
                dist.all_reduce(train_loss, op=dist.ReduceOp.AVG)
            for train_metric in train_metrics:
                train_metric.gather()

            torch.cuda.synchronize()
            step_time_ms = 1000 * (time.perf_counter() - step_time_start)
            train_step_time_ms = step_time_ms - eval_step_time_ms - save_step_time_ms
            train_time_ms += train_step_time_ms

            if self.main_process:
                train_step_avg = train_time_ms / (step + 1)
                print(
                    f"step:{step}/{self.args.max_train_steps} "
                    f"train_loss:{train_loss.item():.4f} "
                    f"train_time:{train_time_ms:.0f}ms "
                    f"train_step_avg:{train_step_avg:.2f}ms"
                )
                if is_log_step:
                    train_batch_tokens = (
                        self.args.train_batch_size * self.args.block_size
                    )
                    train_metric_data = {
                        "train_loss": train_loss.item(),
                        "train_time": train_time_ms,
                        "train_step_avg": train_step_avg,
                        "num_tokens": (step + 1) * train_batch_tokens,
                    }
                    for train_metric in train_metrics:
                        train_metric_data.update(train_metric.to_dict())
                    if not self.args.debug:
                        wandb.log(train_metric_data, step=step)

            torch.cuda.empty_cache()  # Free memory

        self.save_checkpoint(self.args.max_train_steps, revision="main")

        max_memory_allocated_mib = torch.cuda.max_memory_allocated() // 1024 // 1024
        print(f"rank:{get_rank()} max_memory_allocated:{max_memory_allocated_mib}MiB")

        self.destroy_upload_queue()
        if self.is_dist_available:
            dist.destroy_process_group()
        wandb.finish()

    def train_step(
        self,
        train_dataloader: Generator[dict[str, Tensor], None, None],
        autocast: torch.autocast,
        is_log_step: bool,
    ) -> tuple[Tensor, list[TrainerMetric]]:
        train_loss = torch.zeros(1, device=self.device)
        train_metrics = []
        if self.train_metric_cls is not None and is_log_step:
            train_metrics = [
                train_metric_cls(self.args)
                for train_metric_cls in self.train_metric_cls
            ]

        num_gradient_accumulation_steps = self.args.gradient_accumulation_steps
        for gradient_accumulation_step in range(self.args.gradient_accumulation_steps):
            try:
                sync_context = nullcontext()
                if (
                    self.is_dist_available
                    and not gradient_accumulation_step
                    == self.args.gradient_accumulation_steps - 1
                ):
                    sync_context = self.ddp_model.no_sync()

                with sync_context:
                    with autocast:
                        inputs = next(train_dataloader)
                        outputs: CausalLMOutputWithPast = self.ddp_model(
                            input_ids=inputs["input_ids"],
                            labels=inputs["labels"],
                            **(self.args.forward_kwargs or {}),
                        )
                        loss = self.compute_loss(outputs, inputs)
                    loss.backward()  # Outside autocast, inside sync_context!

                train_loss += loss.detach()
                with torch.no_grad():
                    for train_metric in train_metrics:
                        train_metric.update(outputs, inputs)

                del inputs, outputs, loss  # Free memory
            except StopIteration:
                num_gradient_accumulation_steps = gradient_accumulation_step + 1
                break

        train_loss /= num_gradient_accumulation_steps
        return train_loss, train_metrics

    def compute_loss(
        self, outputs: CausalLMOutputWithPast, inputs: dict[str, Tensor]
    ) -> Tensor:
        if self.compute_loss_func is not None:
            return self.compute_loss_func(outputs, inputs)
        if hasattr(outputs, "loss"):
            loss = outputs.loss
            if loss is not None:
                return loss
        raise ValueError("No `compute_loss_func` provided and `outputs.loss` is None.")
