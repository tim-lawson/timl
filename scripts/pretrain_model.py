from transformers import AutoConfig, AutoModelForCausalLM

from timl.data_loader_tokens import batch_loader_tokens
from timl.trainer import Trainer, TrainingArguments


class Args(TrainingArguments):
    model_name: str
    attn_implementation: str | None = None  # For Gemma 3


def pretrain_model(args: Args):
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_config(
        config, attn_implementation=args.attn_implementation
    )

    trainer = Trainer(model, args, batch_loader_tokens, batch_loader_tokens)
    trainer.train()


if __name__ == "__main__":
    pretrain_model(Args(_cli_parse_args=True))  # ty: ignore[missing-argument,unknown-argument]
