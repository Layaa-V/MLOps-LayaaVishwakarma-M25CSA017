# Assignment 5 – DLOps: LoRA Fine-tuning & IBM ART Adversarial Attacks


## Hugging Face Link : 
https://huggingface.co/Layaa-V/dlops-assignment5-weights/tree/main
Contents present in this link - 
- best_optuna_model.pt: The optimal ViT-S model fine-tuned with LoRA on CIFAR-100, discovered during the Optuna hyperparameter search.
- detector_bim_best.pt: The ResNet-34 binary classifier trained to detect Basic Iterative Method (BIM) adversarial images.
- detector_pgd_best.pt: The ResNet-34 binary classifier trained to detect Projected Gradient Descent (PGD) adversarial images.
- resnet18_cifar10_best.pt: The baseline ResNet18 victim model trained from scratch on clean CIFAR-10 images.


## WanDB Links :  
- Question 1 - https://wandb.ai/0201ai211031-iit/DLOps-Ass5-Q1?nw=nwuser0201ai211031
- Question 1 Optuna - https://wandb.ai/0201ai211031-iit/DLOps-Ass5-Q1-Optuna?nw=nwuser0201ai211031
- Question 2 Part (i)  - https://wandb.ai/0201ai211031-iit/DLOps-Ass5-Q2i?nw=nwuser0201ai211031
- Question 2 Part (ii)  - https://wandb.ai/0201ai211031-iit/DLOps-Ass5-Q2ii?nw=nwuser0201ai211031


## Setup

In terminal - 
docker build -t dlops-ass5 .   (Build image)

docker run --gpus '"device=4"' -it -v $(pwd)/weights:/app/weights -v $(pwd)/data:/app/data dlops-ass5 bash (building and running container)


## Q1 – ViT-S LoRA Fine-tuning on CIFAR-100

### Training Commands
(here wdbkey = your wandb key (generated token))
 
- cd Q1
- python train.py --mode all --wandb_key wdbkey   (Run all experiments)

- python train.py --mode baseline --wandb_key wdbkey    (baseline only)

- python train.py --mode lora --rank 4 --alpha 8 --dropout 0.1 --wandb_key wdbkey    (single LoRA config)

- python optuna_search.py --n_trials 20 --wandb_key wdbkey    (Optuna hyperparameter search)

- python test.py --ckpt ../weights/best_optuna_model.pt --lora --rank 4 --alpha 8


### Results Tables

#### Training & Validation (Baseline – No LoRA)

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|---------------|-----------------|-------------------|---------------------|
| 1     | 0.9140        | 0.7245          | 0.7561            | 0.7894              |
| 2     | 0.5861        | 0.7133          | 0.8230            | 0.7958              |
| 3     | 0.5179        | 0.6887          | 0.8415            | 0.8029              |
| 4     | 0.4662        | 0.6906          | 0.8554            | 0.8020              |
| 5     | 0.4323        | 0.6909          | 0.8640            | 0.8053              |
| 6     | 0.4019        | 0.6836          | 0.8736            | 0.8065              |
| 7     | 0.3749        | 0.6805          | 0.8822            | 0.8063              |
| 8     | 0.3518        | 0.6725          | 0.8907            | 0.8092              |
| 9     | 0.3408        | 0.6628          | 0.8928            | 0.8106              |
| 10    | 0.3286        | 0.6598          | 0.8977            | 0.8126              |

#### Training & Validation (Example – LoRA rank=8, alpha=8, dropout=0.1)

| Epoch | Training Loss | Validation Loss | Training Accuracy | Validation Accuracy |
|-------|---------------|-----------------|-------------------|---------------------|
| 1     | 0.5933        | 0.4052          | 0.8355            | 0.8778              |
| 2     | 0.2961        | 0.3836          | 0.9064            | 0.8866              |
| 3     | 0.2198        | 0.3636          | 0.9281            | 0.8928              |
| 4     | 0.1662        | 0.3913          | 0.9438            | 0.8897              |
| 5     | 0.1227        | 0.3986          | 0.9592            | 0.8923              |
| 6     | 0.0869        | 0.3926          | 0.9705            | 0.8968              |
| 7     | 0.0566        | 0.3976          | 0.9822            | 0.9001              |
| 8     | 0.0408        | 0.3913          | 0.9880            | 0.9017              |
| 9     | 0.0321        | 0.3950          | 0.9912            | 0.9025              |
| 10    | 0.0275        | 0.3925          | 0.9933            | 0.9012              |

#### Test Accuracy Summary

| LoRA | Rank | Alpha | Dropout | Overall Test Accuracy | Trainable Parameters |
|------|------|-------|---------|-----------------------|----------------------|
| No   | —    | —     | —       | 0.8126                | 38,500               |
| Yes  | 2    | 2     | 0.1     | 0.8958                | 93,796               |
| Yes  | 2    | 4     | 0.1     | 0.8945                | 93,796               |
| Yes  | 2    | 8     | 0.1     | 0.8984                | 93,796               |
| Yes  | 4    | 2     | 0.1     | 0.8984                | 149,092              |
| Yes  | 4    | 4     | 0.1     | 0.8997                | 149,092              |
| Yes  | 4    | 8     | 0.1     | 0.8983                | 149,092              |
| Yes  | 8    | 2     | 0.1     | 0.9024                | 259,684              |
| Yes  | 8    | 4     | 0.1     | 0.8994                | 259,684              |
| Yes  | 8    | 8     | 0.1     | 0.9025                | 259,684              |

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
|--------|--------------------|
| PGD    | 85.89%             |
| BIM    | 85.19%             |

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
