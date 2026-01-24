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

---

### Key Observations - 

* Across almost all configurations, the **Adam** optimizer consistently outperformed **SGD**.
* Both models performed exceptionally well at a learning rate of 0.001.
* At lower learning rate of 0.0001, SGD struggled while Adam remained stable. 
* Smaller batch sizes (16) generally yielded slightly better results for SGD at low learning rates compared to batch size 32.
---

