## MLOps Assignment 3: End-to-End Hugging Face Classification & Docker Deployment


**Hugging Face Model:** [https://huggingface.co/Layaa-V/FineTuned-Bert-Classifier](https://huggingface.co/Layaa-V/FineTuned-Bert-Classifier)

---


### Model Selection
DistilBERT was selected because it is a distilled version of the full BERT model that retains approximately 97% of BERT's language understanding capabilities while being 60% faster and 40% smaller. This makes it highly optimal for fine-tuning on local hardware and keeps the final Docker container size manageable.

### Training Summary
The model was fine-tuned using the Hugging Face Trainer API on a sampled dataset. The data was tokenized with padding and truncation (max length 512). The model trained for 3 epochs with a learning rate of 5e-5 and a batch size of 10. The trained weights and tokenizer were then successfully pushed to the Hugging Face Hub.

### Evaluation Comparison
The model was evaluated locally immediately after training.
* **Evaluation Accuracy:** 0.5875

---
