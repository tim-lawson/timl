# Based on https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py

from typing import Any

from pydantic_settings import BaseSettings
from transformers.trainer_utils import SchedulerType
from transformers.training_args import OptimizerNames


class TrainingArguments(BaseSettings):
    output_dir: str

    train_file_pattern: str
    eval_file_pattern: str

    max_train_tokens: int | None = None
    max_train_steps: int | None = None
    max_eval_tokens: int | None = None
    max_eval_steps: int | None = None

    block_size: int

    train_batch_size: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int = -1  # set by Trainer

    learning_rate: float
    weight_decay: float = 0.0
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_grad_norm: float = 0.0

    optim: str | OptimizerNames = "adamw_torch_fused"

    lr_scheduler_type: SchedulerType | str
    lr_scheduler_kwargs: dict | None = None
    warmup_ratio: float = 0.0
    warmup_steps: int = 0

    log_every_steps: int = 1
    save_every_steps: int
    eval_every_steps: int

    seed: int = 0
    full_determinism: bool = False

    bf16: bool = True

    wandb_project: str | None = None
    wandb_run_name: str | None = None

    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_token: str | bool | None = None
    hub_private: bool | None = None

    forward_kwargs: dict[str, Any] | None = None

    use_liger_kernel: bool = False
    liger_kernel_kwargs: dict[str, bool] | None = None
    use_cut_cross_entropy: bool = False
    cut_cross_entropy_kwargs: dict[str, Any] | None = None

    debug: bool = False
