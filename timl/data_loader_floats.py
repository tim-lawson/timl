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

file_handler_floats = BinaryFileHandler(
    torch_dtype_to_int={torch.float32: 0},
    numpy_dtype_to_int={np.float32: 0},
)


def file_loader_floats(
    file: Path, batch_size: int
) -> Generator[tuple[Tensor, Tensor], Any, None]:
    rank, world_size, device = dist.get_rank(), dist.get_world_size(), dist.get_device()

    if batch_size % world_size != 0:
        raise ValueError(
            f"batch_size ({batch_size}) must be divisible by world_size ({world_size})."
        )
    local_batch_size = batch_size // world_size

    values = file_handler_floats.read(file)
    if len(values) < batch_size + 1:  # Batch size in units of tokens
        raise ValueError(
            f"Number of values in {file} is smaller than batch_size ({batch_size})."
        )

    position = 0
    while position + batch_size <= len(values):
        buffer = values[position + rank * local_batch_size :][:local_batch_size]
        weights = buffer.to(
            device=device, non_blocking=torch.cuda.is_available()
        )  # No sync on host side

        yield weights
        position += batch_size


def data_loader_floats(
    file_pattern: str, batch_size: int
) -> Generator[Tensor, Any, None]:
    files = [Path(file) for file in sorted(glob(file_pattern))]
    if len(files) == 0:
        raise FileNotFoundError(f"No files found for pattern: {file_pattern}")
    for file in files:
        yield from file_loader_floats(file, batch_size)


def batch_loader_floats(
    file_pattern: str, per_device_batch_size: int, block_size: int
) -> Generator[dict[str, Tensor], Any, None]:
    batch_size = per_device_batch_size * dist.get_world_size() * block_size
    for weights in data_loader_floats(file_pattern, batch_size):
        yield {"weights": weights.view(-1, block_size)}


def get_num_floats(file_pattern: str) -> int:
    return get_num_values(file_handler_floats, file_pattern)
