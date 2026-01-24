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
* On the MNIST dataset, the **RBF** kernel slightly outperformed the **Polynomial** kernel in accuracy, but it required significantly more training time (~60 seconds vs ~42 seconds).
* Accuracy was noticeably lower on **FashionMNIST** (~85-86%) compared to **MNIST** (~96%), indicating that FashionMNIST presents a more complex classification task for classical SVMs.
* Across both datasets, the **Polynomial** kernel was consistently faster to train than the **RBF** kernel while maintaining comparable accuracy.
---


