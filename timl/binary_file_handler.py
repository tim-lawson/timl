# Based on https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt.py

from glob import glob
from pathlib import Path

import numpy as np
import torch


class BinaryFileHandler:
    HEADER_SIZE = 256  # 1024 bytes
    HEADER_TORCH_DTYPE = torch.int32
    HEADER_NUMPY_DTYPE = np.int32
    MAGIC_NUMBER = 20250805
    VERSION = 1

    def __init__(
        self,
        torch_dtype_to_int: dict[torch.dtype, int],
        numpy_dtype_to_int: dict[type[np.generic], int],
    ):
        self.torch_dtype_to_int = torch_dtype_to_int.copy()
        self.numpy_dtype_to_int = numpy_dtype_to_int.copy()
        torch_int_values = set(self.torch_dtype_to_int.values())
        numpy_int_values = set(self.numpy_dtype_to_int.values())
        if torch_int_values != numpy_int_values:
            raise ValueError("PyTorch and NumPy dtype maps must match.")

    @property
    def int_to_torch_dtype(self) -> dict[int, torch.dtype]:
        return {v: k for k, v in self.torch_dtype_to_int.items()}

    @property
    def int_to_numpy_dtype(self) -> dict[int, type[np.generic]]:
        return {v: k for k, v in self.numpy_dtype_to_int.items()}

    def read_header(self, file: Path) -> tuple[torch.dtype, int]:
        header = torch.from_file(
            str(file),
            shared=False,
            size=self.HEADER_SIZE,
            dtype=self.HEADER_TORCH_DTYPE,
        )
        if header[0] != self.MAGIC_NUMBER:
            raise ValueError(
                f"Invalid magic number in {file}. Expected {self.MAGIC_NUMBER}, found {header[0]}."
            )
        if header[1] != self.VERSION:
            raise ValueError(
                f"Invalid version in {file}. Expected {self.VERSION}, found {header[1]}."
            )
        try:
            dtype = self.int_to_torch_dtype[int(header[2])]
        except KeyError:
            raise ValueError(
                f"Invalid dtype identifier in {file}. Expected {self.torch_dtype_to_int}, found {header[2]}."
            )
        num_values = int(header[3])
        return dtype, num_values

    def read(self, file: Path) -> torch.Tensor:
        dtype, num_values = self.read_header(file)
        values = torch.empty(
            num_values, dtype=dtype, pin_memory=torch.cuda.is_available()
        )
        with file.open("rb") as f:
            f.seek(self.HEADER_SIZE * self.HEADER_TORCH_DTYPE.itemsize)  # skip header
            num_bytes = f.readinto(values.numpy())
        if num_bytes != num_values * dtype.itemsize:
            raise IOError(
                f"Number of values in {file} does not match header. Expected {num_values}, found {num_bytes // dtype.itemsize}."
            )
        return values

    def write(self, file: Path, values: np.ndarray) -> None:
        header = np.zeros(self.HEADER_SIZE, dtype=self.HEADER_NUMPY_DTYPE)
        header[0] = self.MAGIC_NUMBER
        header[1] = self.VERSION
        header[2] = self.numpy_dtype_to_int[values.dtype.type]
        header[3] = len(values)
        with file.open("wb") as f:
            f.write(header.tobytes())
            f.write(values.tobytes())


def get_num_values(file_handler: BinaryFileHandler, file_pattern: str) -> int:
    num_values = 0
    for file in [Path(file) for file in sorted(glob(file_pattern))]:
        num_values += file_handler.read_header(file)[1]
    return num_values
