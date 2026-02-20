import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer
from sklearn.metrics import accuracy_score, classification_report

# --- Configuration ---
# Pulls the repo ID from the environment, defaulting to your repo if not set
REPO_ID = os.environ.get("MODEL_REPO_ID","Layaa-V/FineTuned-Bert-Classifier" )

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

def main():
    print(f"Downloading model {REPO_ID} from Hugging Face Hub...")
    model = AutoModelForSequenceClassification.from_pretrained(REPO_ID)
    
    print("Loading test data...")
    test_dataset = torch.load('test_dataset.pt', weights_only=False)
    
    with open('id2label.json', 'r') as f:
        id2label_str = json.load(f)
        # Convert string keys back to integers for mapping
        id2label = {int(k): v for k, v in id2label_str.items()}
        
    with open('test_labels.json', 'r') as f:
        test_labels = json.load(f)

    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics
    )

    print("Running evaluation...")
    eval_results = trainer.evaluate(eval_dataset=test_dataset)
    print(f"\nEvaluation Results: {eval_results}\n")

    print("Generating Classification Report...")
    predicted_results = trainer.predict(test_dataset)
    predicted_labels_encoded = predicted_results.predictions.argmax(-1).flatten().tolist()
    predicted_labels = [id2label[l] for l in predicted_labels_encoded]

    print(classification_report(test_labels, predicted_labels))

if __name__ == "__main__":
    main()
