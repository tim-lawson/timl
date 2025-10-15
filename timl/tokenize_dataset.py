# Based on https://github.com/KellerJordan/modded-nanogpt/blob/master/data/fineweb.py

import os
from pathlib import Path

import numpy as np
from datasets import load_dataset
from multiprocess.pool import Pool
from tqdm import tqdm

from timl.data_loader_tokens import file_handler_tokens


def tokenize_dataset(
    cache_dir: str | os.PathLike,
    dataset_path: str,
    dataset_name: str | None = None,
    dataset_split: str = "train",
    dataset_text_key: str = "text",
    tokenizer_name: str | None = None,
    encoding_name: str | None = None,
    tokenizer_path: str | None = None,
    token_dtype: type[np.integer] = np.uint16,
    num_tokens_per_file: int = 2**27,  # 134M
    min_file_index: int = 0,
    max_file_index: int | None = None,
    num_proc: int | None = None,
    chunksize: int = 4096,  # 'Benchmarked' with 16 CPUs
    streaming: bool = False,
    trust_remote_code: bool = False,
    eot_token: str = "<eot>",
):
    if tokenizer_name is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        eot_token = getattr(
            tokenizer, "eot_token", None
        ) or tokenizer.special_tokens_map.get("eot_token")
        eot_token = (
            eot_token
            or tokenizer.eos_token
            or tokenizer.special_tokens_map.get("eot_token")
        )
        if eot_token is None:
            raise ValueError(
                f"Tokenizer {tokenizer_name} has no eot_token or eos_token."
            )
        eot_token_id = tokenizer.convert_tokens_to_ids(eot_token)
        vocab_size = tokenizer.vocab_size

        def encode(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False)

    elif encoding_name is not None:
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        eot_token_id = encoding.eot_token
        vocab_size = encoding.n_vocab

        def encode(text: str) -> list[int]:
            return encoding.encode_ordinary(text)

    elif tokenizer_path is not None:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(tokenizer_path)
        eot_token_id = tokenizer.token_to_id(eot_token)
        if eot_token_id is None:
            raise ValueError(f"Tokenizer {tokenizer_path} has no <eot> token.")
        vocab_size = tokenizer.get_vocab_size()

        def encode(text: str) -> list[int]:
            return tokenizer.encode(text, add_special_tokens=False).ids

    else:
        raise ValueError(
            "Either tokenizer_name, encoding_name, or tokenizer_path must be provided."
        )

    if vocab_size > np.iinfo(token_dtype).max:
        raise ValueError(
            f"Vocabulary size {vocab_size} exceeds maximum for dtype {token_dtype}."
        )

    def tokenize(row: dict) -> np.ndarray:
        text = row.get(dataset_text_key)
        if not isinstance(text, str):
            return np.array([], dtype=token_dtype)
        return np.array([eot_token_id] + encode(text), dtype=token_dtype)

    num_proc = num_proc or os.cpu_count() or 1
    num_proc = max(1, num_proc - 1)  # Leave one CPU core free

    dataset = load_dataset(
        path=dataset_path,
        name=dataset_name,
        split=dataset_split,
        num_proc=None if streaming else num_proc,
        streaming=streaming,
        trust_remote_code=trust_remote_code,
    )

    cache_dir = Path(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    with Pool(processes=num_proc) as pool:
        file_index = 0
        buffer = np.empty(num_tokens_per_file, dtype=token_dtype)
        buffer_position = 0
        progress_bar: tqdm | None = None

        for tokens in pool.imap(tokenize, dataset, chunksize=chunksize):
            if max_file_index is not None and file_index > max_file_index:
                break

            num_tokens = len(tokens)

            if buffer_position + num_tokens < num_tokens_per_file:
                buffer[buffer_position : buffer_position + num_tokens] = tokens
                buffer_position += num_tokens

                if progress_bar is None:
                    progress_bar = tqdm(
                        desc=f"file {file_index}",
                        total=num_tokens_per_file,
                        unit="tokens",
                    )
                progress_bar.update(num_tokens)

            else:
                num_tokens_remain = num_tokens_per_file - buffer_position
                progress_bar.update(num_tokens_remain)  # ty: ignore[unresolved-attribute]
                buffer[buffer_position : buffer_position + num_tokens_remain] = tokens[
                    :num_tokens_remain
                ]

                if file_index >= min_file_index:
                    file_handler_tokens.write(
                        cache_dir / f"{dataset_split}_{file_index:06d}.bin", buffer
                    )

                file_index += 1
                progress_bar = None

                buffer[0 : num_tokens - num_tokens_remain] = tokens[num_tokens_remain:]
                buffer_position = num_tokens - num_tokens_remain

        if file_index >= min_file_index and buffer_position > 0:
            if max_file_index is not None and file_index > max_file_index:
                return
            file_handler_tokens.write(
                cache_dir / f"{dataset_split}_{file_index:06d}.bin",
                buffer[:buffer_position],
            )
