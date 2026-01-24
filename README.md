# ML-DL-Ops Lab Assignment 1 Results

## Q1(a): ResNet Performance Analysis on MNIST/FashionMNIST
The following table summarizes the classification test accuracy for **ResNet-18** and **ResNet-50** across various hyperparameters, including Batch Size, Optimizer, and Learning Rate.

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


### Key Observations - 

* Across almost all configurations, the **Adam** optimizer consistently outperformed **SGD**.
* Both models performed exceptionally well at a learning rate of 0.001.
* At lower learning rate of 0.0001, SGD struggled while Adam remained stable. 
* Smaller batch sizes (16) generally yielded slightly better results for SGD at low learning rates compared to batch size 32.
---

## Q1(b): SVM Performance
This section reports the performance of a **Support Vector Machine (SVM)** classifier using **Polynomial** and **RBF** kernels on MNIST and FashionMNIST datasets.

| Dataset | Kernel | Test Accuracy (%) | Train Time (ms) |
|:---|:---:|:---:|:---:|
| MNIST | poly | 96.41 | 42719.35 |
| MNIST | rbf | 96.45 | 60662.33 |
| FashionMNIST | poly | 86.19 | 47858.95 |
| FashionMNIST | rbf | 85.16 | 58355.74 |

### Key Observations - 
* On the MNIST dataset, the **RBF** kernel slightly outperformed the **Polynomial** kernel in accuracy, but it required significantly more training time (nearly 60 seconds vs nearly 42 seconds).
* Accuracy was noticeably lower on **FashionMNIST** (85-86%) compared to **MNIST** (96%), indicating that FashionMNIST presents a more complex classification task for classical SVMs.
* Across both datasets, the **Polynomial** kernel was consistently faster to train than the **RBF** kernel while maintaining comparable accuracy.
---

## Q2: Hardware & Computational Efficiency (FashionMNIST)
This section analyzes the performance of ResNet architectures across different compute devices (**CPU vs. GPU**), measuring accuracy, training time, and computational complexity (FLOPs).

### Hardware Comparison & FLOPs Table
| Compute | Batch Size | Optimizer | LR | Model | Test Accuracy (%) | Train Time (ms) | FLOPs |
|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---:|
| GPU | 16 | SGD | 0.001 | ResNet-18 | 87.64 | 2239.42 | 142,441,984 |
| GPU | 16 | Adam | 0.001 | ResNet-18 | 88.65 | 2287.93 | 142,441,984 |
| GPU | 16 | SGD | 0.001 | ResNet-50 | 80.92 | 4807.82 | 330,881,024 |
| GPU | 16 | Adam | 0.001 | ResNet-50 | 84.77 | 4548.97 | 330,881,024 |
| CPU | 16 | SGD | 0.001 | ResNet-18 | 87.77 | 9764.73 | 142,441,984 |
| CPU | 16 | Adam | 0.001 | ResNet-18 | 89.48 | 11360.59 | 142,441,984 |
| CPU | 16 | SGD | 0.001 | ResNet-50 | 78.84 | 18040.84 | 330,881,024 |
| CPU | 16 | Adam | 0.001 | ResNet-50 | 74.02 | 21231.78 | 330,881,024 |

### Analysis of Q2 Results
* **Hardware Acceleration**: The GPU provided a massive speedup compared to the CPU. For ResNet-50, training time dropped from 21,231 ms on CPU to 4,548 ms on GPU, a reduction of approximately 78%. 
* **FLOPs vs. Depth**: **ResNet-50** requires significantly higher computational power compared to **ResNet-18**. On FashionMNIST, the higher FLOP count of ResNet-50 did not translate to higher accuracy, suggesting ResNet-18 is more efficient for this specific task.
* **Compute Consistency**: Accuracy remained largely consistent across CPU and GPU for ResNet-18. However, ResNet-50 showed more variance on CPU, likely due to the extreme training time affecting convergence stability during the test window.

---


