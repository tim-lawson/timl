from typing import Protocol

import torch
import torch.distributed as dist
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from timl.distributed import get_device
from timl.training_args import TrainingArguments


# Implementations are responsible for gathering metrics across processes!
class TrainerMetric(Protocol):
    def __init__(self, args: TrainingArguments) -> None: ...

    def update(
        self, outputs: CausalLMOutputWithPast, inputs: dict[str, Tensor]
    ) -> None: ...

    def gather(self) -> None: ...

    def to_dict(self) -> dict[str, float]: ...


# For example...
def create_loss_metric_cls(key: str) -> type[TrainerMetric]:
    class loss_metric_cls(TrainerMetric):
        def __init__(self, args):
            self.loss = torch.zeros(1, device=get_device())
            self.steps = 0

        def update(self, outputs, inputs):
            if outputs.loss is not None:  # Mean in case we've used reduction="none"
                self.loss += outputs.loss.mean().detach()
            self.steps += 1

        def gather(self):
            dist.all_reduce(self.loss, op=dist.ReduceOp.AVG)
            self.loss /= self.steps

        def to_dict(self):
            return {key: self.loss.item()}

    return loss_metric_cls
