# Based on https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py#L528-L550

from collections.abc import Generator
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

import timl.distributed as dist
from timl.binary_file_handler import BinaryFileHandler, get_num_values

file_handler_tokens = BinaryFileHandler(
    torch_dtype_to_int={torch.uint16: 0, torch.uint32: 1},
    numpy_dtype_to_int={np.uint16: 0, np.uint32: 1},
)


def file_loader_tokens(
    file: Path, batch_size: int, ignore_index: int = -100
) -> Generator[tuple[Tensor, Tensor], Any, None]:
    rank, world_size, device, cuda = (
        dist.get_rank(),
        dist.get_world_size(),
        dist.get_device(),
        torch.cuda.is_available(),
    )

    if batch_size % world_size != 0:
        raise ValueError(
            f"Batch size ({batch_size}) must be divisible by world size ({world_size})."
        )
    local_batch_size = batch_size // world_size

    values = file_handler_tokens.read(file)
    if len(values) < batch_size + 1:  # Batch size in units of tokens
        raise ValueError(
            f"Number of tokens in {file} is smaller than batch size ({batch_size})."
        )

    # If len(tokens) is divisible by batch_size, we return the last batch but ignore its last token.
    # We replace the last token with ignore_index later, but tokens is uint16 or uint32 here.
    padding_token = torch.tensor([0], dtype=values.dtype, device=values.device)
    values = torch.cat([values, padding_token], dim=0)

    position = 0
    while position + batch_size + 1 <= len(values):
        buffer = values[position + rank * local_batch_size :][: local_batch_size + 1]
        inputs = buffer[:-1].to(
            device=device, dtype=torch.int32, non_blocking=cuda
        )  # No sync on host side
        labels = buffer[1:].to(
            device=device, dtype=torch.int64, non_blocking=cuda
        )  # Cross-entropy loss expects int64

        # If this is the last batch, replace the last token with ignore_index.
        if position + batch_size + 1 == len(values):
            labels[-1] = ignore_index

        yield inputs, labels
        position += batch_size


def data_loader_tokens(
    file_pattern: str, batch_size: int, ignore_index: int = -100
) -> Generator[tuple[Tensor, Tensor], Any, None]:
    files = [Path(file) for file in sorted(glob(file_pattern))]
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern}")
    for file in files:
        yield from file_loader_tokens(file, batch_size, ignore_index)


def batch_loader_tokens(
    file_pattern: str,
    per_device_batch_size: int,
    block_size: int,
    ignore_index: int = -100,
) -> Generator[dict[str, Tensor], Any, None]:
    batch_size = per_device_batch_size * dist.get_world_size() * block_size
    for input_ids, labels in data_loader_tokens(file_pattern, batch_size, ignore_index):
        yield {
            "input_ids": input_ids.view(-1, block_size),
            "labels": labels.view(-1, block_size),
        }


def get_num_tokens(file_pattern: str) -> int:
    return get_num_values(file_handler_tokens, file_pattern)
