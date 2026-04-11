"""
Push best LoRA model weights to HuggingFace Hub.
Usage:
    python push_to_hub.py \
        --ckpt weights/best_optuna_model.pt \
        --rank 4 --alpha 8 --dropout 0.1 \
        --hf_token YOUR_HF_TOKEN \
        --repo_id YOUR_USERNAME/dlops-ass5-vit-lora-cifar100
"""

import argparse
import os
import torch
from transformers import ViTForImageClassification
from peft import LoraConfig, get_peft_model, TaskType
from huggingface_hub import HfApi, create_repo

NUM_CLASSES = 100


def build_model(rank, alpha, dropout):
    base = ViTForImageClassification.from_pretrained(
        "WinKawaks/vit-small-patch16-224",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )
    cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=["query", "key", "value"],
        bias="none",
    )
    return get_peft_model(base, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",     required=True)
    parser.add_argument("--rank",     type=int,   default=4)
    parser.add_argument("--alpha",    type=int,   default=8)
    parser.add_argument("--dropout",  type=float, default=0.1)
    parser.add_argument("--hf_token", required=True)
    parser.add_argument("--repo_id",  required=True,
                        help="e.g. username/dlops-ass5-vit-lora-cifar100")
    args = parser.parse_args()

    # Load model
    model = build_model(args.rank, args.alpha, args.dropout)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    print(f"Loaded checkpoint from {args.ckpt}")

    # Create repo (if not exists)
    api = HfApi(token=args.hf_token)
    try:
        create_repo(args.repo_id, token=args.hf_token, exist_ok=True)
    except Exception as e:
        print(f"Repo creation: {e}")

    # Push model card + weights
    model.push_to_hub(
        args.repo_id,
        token=args.hf_token,
        commit_message="Upload best LoRA ViT-S model (Assignment 5)",
    )

    # Also upload the raw .pt file for reproducibility
    api.upload_file(
        path_or_fileobj=args.ckpt,
        path_in_repo=os.path.basename(args.ckpt),
        repo_id=args.repo_id,
        repo_type="model",
        token=args.hf_token,
    )
    print(f"Model pushed to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
