from typing import Literal

import numpy as np
from pydantic_settings import BaseSettings

from timl.tokenize_dataset import tokenize_dataset


class Args(BaseSettings):
    cache_dir: str
    dataset_path: str
    dataset_name: str | None = None
    dataset_split: str = "train"
    dataset_text_key: str = "text"
    tokenizer_name: str | None = None
    encoding_name: str | None = None
    tokenizer_path: str | None = None
    token_dtype: Literal["uint16", "uint32"] = "uint16"
    num_tokens_per_file: int = 2**27  # 134M
    min_file_index: int = 0
    max_file_index: int | None = None
    num_proc: int | None = None
    chunksize: int = 4096  # 'Benchmarked' with 16 CPUs
    streaming: bool = False
    trust_remote_code: bool = False
    eot_token: str = "<eot>"


if __name__ == "__main__":
    args = Args(_cli_parse_args=True)  # ty: ignore[missing-argument,unknown-argument]
    tokenize_dataset(
        cache_dir=args.cache_dir,
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        dataset_text_key=args.dataset_text_key,
        tokenizer_name=args.tokenizer_name,
        encoding_name=args.encoding_name,
        tokenizer_path=args.tokenizer_path,
        token_dtype=np.uint32 if args.token_dtype == "uint32" else np.uint16,
        num_tokens_per_file=args.num_tokens_per_file,
        min_file_index=args.min_file_index,
        max_file_index=args.max_file_index,
        num_proc=args.num_proc,
        chunksize=args.chunksize,
        streaming=args.streaming,
        trust_remote_code=args.trust_remote_code,
        eot_token=args.eot_token,
    )
