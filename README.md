# DL-Ops Lab Assignment 1 Results

## Q1(a): ResNet Performance Analysis on MNIST/FashionMNIST
The following table summarizes the classification test accuracy for **ResNet-18** and **ResNet-50** across various hyperparameters, including Batch Size, Optimizer, and Learning Rate.

### Classification Test Accuracy Table
| Batch Size | Optimizer | Learning Rate | ResNet-18 Accuracy (%) | ResNet-50 Accuracy (%) |
|:---:|:---:|:---:|:---:|:---:|
| 16 | SGD | 0.001 | 98.87 | 98.59 |
| 16 | SGD | 0.0001 | 95.27 | 89.31 |
| 16 | Adam | 0.001 | 99.39 | 99.18 |
| 16 | Adam | 0.0001 | 99.13 | 99.00 |
| 32 | SGD | 0.001 | 98.34 | 97.98 |
| 32 | SGD | 0.0001 | 92.51 | 67.95 |
| 32 | Adam | 0.001 | 99.36 | 99.07 |
| 32 | Adam | 0.0001 | 99.10 | 97.54 |

---

### Detailed Analysis of Q1(a) Results

Based on the experimental data provided in the table, several key observations can be made regarding the impact of hyperparameters on model performance:

#### 1. Optimizer Efficiency: Adam vs. SGD
* [cite_start]**Adam Superiority**: Across almost all configurations, the **Adam** optimizer consistently outperformed **SGD**[cite: 23]. Adam achieved accuracies >99% in most cases, demonstrating its robustness and faster convergence on these datasets.
* [cite_start]**SGD Sensitivity**: SGD showed significant performance drops when the learning rate was reduced to **0.0001**, particularly with the larger ResNet-50 model[cite: 23].

#### 2. Learning Rate Impact
* [cite_start]**High Learning Rate (0.001)**: Both models performed exceptionally well at a learning rate of 0.001[cite: 23].
* **Low Learning Rate (0.0001)**: While Adam remained stable, SGD struggled at this lower rate. [cite_start]The most drastic drop occurred with **ResNet-50** at Batch Size 32, where accuracy fell to **67.95%**, likely due to the model getting stuck in local minima or failing to converge within the epoch limit[cite: 23].

#### 3. Model Depth: ResNet-18 vs. ResNet-50
* [cite_start]**ResNet-18 Stability**: ResNet-18 proved to be more stable and easier to train on these datasets[cite: 22, 23]. It maintained high accuracy (above 92%) across all tests.
* [cite_start]**ResNet-50 Complexity**: While ResNet-50 is a more powerful architecture, it showed higher sensitivity to hyperparameter changes[cite: 22, 23]. This suggests that for simpler datasets like MNIST or FashionMNIST, the increased depth of ResNet-50 can make optimization more challenging without specific tuning.

#### 4. Batch Size Observations
* [cite_start]Smaller batch sizes (16) generally yielded slightly better results for SGD at low learning rates compared to batch size 32[cite: 23].
* [cite_start]For Adam, the batch size had a negligible effect on accuracy, maintaining near-perfect scores in both scenarios[cite: 23].

---

