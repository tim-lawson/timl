import os

import torch
import torch.distributed as dist


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", get_local_rank())
    return torch.device("cpu")


def is_main_process() -> bool:
    return os.environ.get("RANK", "0") == "0"


def is_dist_available() -> bool:
    return all(var in os.environ for var in ["RANK", "LOCAL_RANK", "WORLD_SIZE"])


def init_distributed() -> tuple[int, int, int, torch.device, bool]:
    if is_dist_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    rank = get_rank()
    world_size = get_world_size()
    device = get_device()

    return rank, local_rank, world_size, device, is_main_process()


def print0(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)
