## MLOps Assignment 3: End-to-End Hugging Face Classification & Docker Deployment


**Hugging Face Model:** [https://huggingface.co/Layaa-V/FineTuned-Bert-Classifier](https://huggingface.co/Layaa-V/FineTuned-Bert-Classifier)

---
## Repository file details 

* **`train.py`**: Handles data downloading, preprocessing, fine-tuning the DistilBERT model, and pushing the final artifacts to the Hugging Face Hub.
* **`eval.py`**: Pulls the fine-tuned model from Hugging Face and evaluates its performance against the saved test dataset.
* **`requirements.txt`**: Lists all the necessary Python library dependencies required to execute the training and evaluation scripts.
* **`Dockerfile`**: The blueprint for the heavy, GPU-enabled development environment used to train the model locally.
* **`Dockerfile.eval`**: A lightweight, production-ready blueprint used exclusively to spin up a container and evaluate the model from the cloud.
* **`.gitignore`**: Prevents heavy PyTorch model weights, raw datasets, and cache files from being tracked and pushed to GitHub.

### Model Selection
DistilBERT was selected because it is a distilled version of the full BERT model that retains approximately 97% of BERT's language understanding capabilities while being 60% faster and 40% smaller. This makes it highly optimal for fine-tuning on local hardware and keeps the final Docker container size manageable.

### Training Summary
The model was fine-tuned using the Hugging Face Trainer API on a sampled dataset. The data was tokenized with padding and truncation (max length 512). The model trained for 3 epochs with a learning rate of 5e-5 and a batch size of 10. The trained weights and tokenizer were then successfully pushed to the Hugging Face Hub.

### Evaluation Comparison
The model was evaluated locally immediately after training.
* **Evaluation Accuracy:** 0.5875

---
