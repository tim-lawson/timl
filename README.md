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

### Roadmap

- Support multiple optimizers (and LR schedulers)
- Implement Muon (with auxiliary AdamW) as a standard optimizer
- Forward logging configuration to upload worker process?
- Support custom models (with compatible outputs)
