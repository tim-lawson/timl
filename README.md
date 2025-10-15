# timl

A lightweight framework for pre-training Transformer language models in PyTorch.
Designed for small-scale, academic research projects with data parallelism (DDP) and mixed-precision (`bfloat16`) on NVIDIA GPUs.
The API is based on [Transformers' Trainer](https://huggingface.co/docs/transformers/trainer); the data format and training code are based on the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).

### Features

In 'chronological' order, for now:

- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) Triton kernels (like Transformers)
- [Cut Cross-Entropy (CCE)](https://github.com/apple/ml-cross-entropy) for reduced memory footprint
- Simple data parallelism (DDP) — just do/don't use `torchrun` to launch
- Customisable loss functions (like Transformers)
- Customisable metrics at evaluation (like Transformers) **and** training
- Custom data-loader interface (designed for efficient binary formats)
  - Token-based limits on training and evaluation steps
- Weights & Biases integration as standard
- Non-blocking checkpoint uploads to Hugging Face as revisions (e.g., `step0`)

### Setup

I recommend using `uv`.
To install the core dependencies:

```bash
uv sync
```

To install optional dependencies:

```bash
uv sync --extra cut-cross-entropy
uv sync --extra flash-attn
uv sync --extra liger-kernel
uv sync --all-extras
```

### Usage

This repository can be used as a standalone tool to tokenize a dataset and pre-train a model from scratch.
The `timl` directory can also be imported as a library to build custom training scripts (examples coming soon).

#### 1\. Tokenize a dataset

First, use the `tokenize_dataset.py` script to convert a text dataset from the Hugging Face Hub into the custom binary format used for training.
For example, to generate two files (each containing $2^{27}$ tokens) from the English subset of C4 using the GPT-2 tiktoken tokenizer:

```bash
python scripts/tokenize_dataset.py --cache_dir data/gpt2_c4 --dataset_path allenai/c4 --dataset_name en --encoding_name gpt2 --max_file_index 1 --streaming true
```

Optionally, use the first file as a validation split:

```bash
mv data/gpt2_c4/train_000000.bin data/gpt2_c4/validation_000000.bin
```

For all available arguments, please refer to `scripts/tokenize_dataset.py`.

#### 2\. Pre-train a model

Next, use the `pretrain_model.py` script to train a model from Hugging Face Transformers on the pre-tokenized data. This script uses a `TrainingArguments` class which you can populate via command-line arguments.

```bash
torchrun --standalone --nproc_per_node=1 scripts/pretrain_model.py --model_name gpt2 --output_dir models/gpt2_c4 --train_file_pattern "data/gpt2_c4/train_*.bin" --eval_file_pattern "data/gpt2_c4/validation_*.bin" --max_eval_tokens 10485760 --block_size 1024 --train_batch_size 512 --per_device_train_batch_size 32 --per_device_eval_batch_size 64 --learning_rate 6e-4 --weight_decay 0.1 --adam_beta1 0.8 --adam_beta2 0.95 --adam_epsilon 1e-8 --max_grad_norm 1.0 --optim adamw_torch_fused --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr": 0.00001}' --warmup_ratio 0.01 --log_every_steps 1 --save_every_steps 1000 --eval_every_steps 100 --wandb_project timl --wandb_run_name gpt2_c4 --push_to_hub true --hub_model_id tim-lawson/gpt2_c4_example --debug false
```

For all available arguments, please refer to `timl/training_args.py`.
Most arguments behave the same as Transformers' `TrainingArguments` ([documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments)).

### Roadmap

- Support multiple optimizers (and LR schedulers)
- Implement Muon (with auxiliary AdamW) as a standard optimizer
- Forward logging configuration to upload worker process?
- Support custom models (with compatible outputs)
