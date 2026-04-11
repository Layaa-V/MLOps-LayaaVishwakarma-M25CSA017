# Assignment 5 – DLOps: LoRA Fine-tuning & IBM ART Adversarial Attacks

> **Course:** Deep Learning Operations (DLOps)  
> **Deadline:** 03/04/2026 11:59 PM  
> **WandB Project:** [Add your WandB link here]  
> **HuggingFace Model:** [Add your HuggingFace link here]

-----

## Table of Contents

1.  [Project Structure](https://www.google.com/search?q=%23project-structure)
2.  [Setup](https://www.google.com/search?q=%23setup)
3.  [Q1 – ViT-S LoRA Fine-tuning on CIFAR-100](https://www.google.com/search?q=%23q1--vit-s-lora-fine-tuning-on-cifar-100)
4.  [Q2 – Adversarial Attacks with IBM ART](https://www.google.com/search?q=%23q2--adversarial-attacks-with-ibm-art)
5.  [Results](https://www.google.com/search?q=%23results)

-----

## Project Structure

```
Assignment5/
├── Dockerfile
├── requirements.txt
├── README.md
├── Q1/
│   ├── train.py           # Main training script (baseline + LoRA grid)
│   ├── optuna_search.py   # Optuna hyperparameter search
│   └── test.py            # Evaluation + per-class histogram
├── Q2/
│   ├── fgsm.py            # Q2(i): FGSM from scratch vs IBM ART
│   └── detect.py          # Q2(ii): Adversarial detection with PGD & BIM
└── weights/               # Saved model checkpoints (pushed to GitHub)
```

-----

## Setup

### Option A – Docker (Recommended, as required by the assignment)

```bash
# Build image
docker build -t dlops-ass5 .

# Run interactive container (mount current dir for persistent weights)
docker run --gpus all -it \
  -v $(pwd)/weights:/app/weights \
  -v $(pwd)/data:/app/data \
  dlops-ass5 bash
```

### Option B – Local (pip)

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

-----

## Q1 – ViT-S LoRA Fine-tuning on CIFAR-100

### Overview

  - **Model:** `WinKawaks/vit-small-patch16-224` (ViT-S pretrained on ImageNet-21k)
  - **Dataset:** CIFAR-100
  - **Experiments:** Baseline (head-only) + LoRA grid (rank ∈ {2,4,8}, alpha ∈ {2,4,8}, dropout=0.1)
  - **LoRA target modules:** `query`, `key`, `value`

### Training Commands

```bash
# 1. Run all experiments (baseline + full LoRA grid)
cd Q1
python train.py --mode all --wandb_key YOUR_WANDB_KEY

# 2. Run baseline only
python train.py --mode baseline --wandb_key YOUR_WANDB_KEY

# 3. Run a single LoRA config
python train.py --mode lora --rank 4 --alpha 8 --dropout 0.1 --wandb_key YOUR_WANDB_KEY

# 4. Optuna hyperparameter search (20 trials by default)
python optuna_search.py --n_trials 20 --wandb_key YOUR_WANDB_KEY

# 5. Test / evaluate a saved checkpoint
python test.py --ckpt ../weights/best_optuna_model.pt --lora --rank 4 --alpha 8
```

### Results Tables

#### Training & Validation (Example – LoRA rank=8, alpha=8, dropout=0.1)

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc |
|-------|-----------|---------|-----------|---------|
| 1     | -         | -       | -         | -       |
| 2     | -         | -       | -         | -       |
| 3     | -         | -       | -         | -       |
| 4     | -         | -       | -         | -       |
| 5     | -         | -       | -         | -       |
| 6     | -         | -       | -         | -       |
| 7     | -         | -       | -         | -       |
| 8     | -         | -       | -         | -       |
| 9     | -         | -       | -         | -       |
| 10    | 0.02746   | 0.39249 | 0.99326   | 0.9012  |

#### Test Accuracy Summary

| LoRA | Rank | Alpha | Dropout | Overall Test Acc | Trainable Params |
|------|------|-------|---------|-----------------|-----------------|
| No   | —    | —     | —       | 0.8126          | 38,500          |
| Yes  | 2    | 2     | 0.1     | 0.8952          | 93,796          |
| Yes  | 2    | 4     | 0.1     | 0.8942          | 93,796          |
| Yes  | 2    | 8     | 0.1     | 0.8984          | 93,796          |
| Yes  | 4    | 2     | 0.1     | 0.8984          | 149,092         |
| Yes  | 4    | 4     | 0.1     | 0.8997          | 149,092         |
| Yes  | 4    | 8     | 0.1     | 0.8975          | 149,092         |
| Yes  | 8    | 2     | 0.1     | 0.9013          | 259,684         |
| Yes  | 8    | 4     | 0.1     | 0.8994          | 259,684         |
| Yes  | 8    | 8     | 0.1     | 0.9012          | 259,684         |

-----

## Q2 – Adversarial Attacks with IBM ART

### Q2(i) – FGSM: From Scratch vs IBM ART

**Dataset:** CIFAR-10  
**Model:** ResNet18 trained from scratch (target ≥ 72% clean accuracy)

```bash
cd Q2

# Step 1: Train ResNet18 from scratch
python fgsm.py --train --epochs 50 --wandb_key YOUR_WANDB_KEY

# Step 2: Evaluate FGSM attacks (skips training if checkpoint exists)
python fgsm.py --ckpt ../weights/resnet18_cifar10_best.pt --wandb_key YOUR_WANDB_KEY
```

**Metrics logged to WandB:**

  - Clean vs adversarial accuracy at ε ∈ {0.01, 0.02, 0.05, 0.1, 0.2}
  - 10 visual samples: Original / FGSM-Scratch / FGSM-ART comparison grid

### Q2(ii) – Adversarial Detection (PGD & BIM)

**Detector Model:** ResNet-34 binary classifier (clean=0, adversarial=1)

```bash
# Train detector for PGD and BIM attacks
python detect.py \
  --victim_ckpt ../weights/resnet18_cifar10_best.pt \
  --eps 0.03 \
  --epochs 20 \
  --wandb_key YOUR_WANDB_KEY
```

**Expected Results:**

  - Detection accuracy ≥ 70% for both PGD and BIM
  - 10 clean + 10 adversarial samples logged to WandB per attack

#### Detection Results Summary

| Attack | Detection Accuracy |
|--------|-------------------|
| PGD    | 85.89%            |
| BIM    | 85.19%            |

-----

## WandB & HuggingFace Links

| Resource | Link |
|----------|------|
| WandB Q1 (LoRA grid) | [Add link] |
| WandB Q1 (Optuna) | [Add link] |
| WandB Q2(i) FGSM | [Add link] |
| WandB Q2(ii) Detection | [Add link] |
| HuggingFace best model | [Add link] |

-----

## Pushing Best Model to HuggingFace

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="weights/best_optuna_model.pt",
    path_in_repo="best_optuna_model.pt",
    repo_id="YOUR_HF_USERNAME/dlops-ass5-vit-lora",
    repo_type="model",
    token="YOUR_HF_TOKEN",
)
```
