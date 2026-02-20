import os
import json
import random
import gzip
import requests
import torch
import pickle
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score

# --- Configuration ---
MODEL_NAME = 'distilbert-base-cased'
MAX_LENGTH = 512
REPO_ID = "Layaa-V/FineTuned-Bert-Classifier"

# --- Data Loading ---
genre_url_dict = {
    'poetry': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz',
    'children': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz',
    'comics_graphic': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz',
    'fantasy_paranormal': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz',
    'history_biography': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz',
    'mystery_thriller_crime': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz',
    'romance': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz',
    'young_adult': 'https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz'
}

def load_reviews(url, head=10000, sample_size=2000):
    reviews = []
    count = 0
    response = requests.get(url, stream=True)
    with gzip.open(response.raw, 'rt', encoding='utf-8') as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d['review_text'])
            count += 1
            if head is not None and count >= head:
                break
    return random.sample(reviews, min(sample_size, len(reviews)))

print("Downloading and sampling data...")
genre_reviews_dict = {}
for genre, url in genre_url_dict.items():
    genre_reviews_dict[genre] = load_reviews(url, head=10000, sample_size=1000)

train_texts, train_labels = [], []
test_texts, test_labels = [], []

for _genre, _reviews in genre_reviews_dict.items():
    for _review in _reviews[:800]:
        train_texts.append(_review)
        train_labels.append(_genre)
    for _review in _reviews[800:]:
        test_texts.append(_review)
        test_labels.append(_genre)

# --- Encoding ---
print("Encoding data...")
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

unique_labels = set(label for label in train_labels)
label2id = {label: id for id, label in enumerate(unique_labels)}
id2label = {id: label for label, id in label2id.items()}

# Save id2label for the eval script
with open('id2label.json', 'w') as f:
    json.dump(id2label, f)
# Save test_labels for classification report in eval script
with open('test_labels.json', 'w') as f:
    json.dump(test_labels, f)

train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=MAX_LENGTH)
test_encodings  = tokenizer(test_texts, truncation=True, padding=True, max_length=MAX_LENGTH)

train_labels_encoded = [label2id[y] for y in train_labels]
test_labels_encoded  = [label2id[y] for y in test_labels]

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

train_dataset = MyDataset(train_encodings, train_labels_encoded)
test_dataset = MyDataset(test_encodings, test_labels_encoded)

# Save test dataset so eval.py can load it without redownloading everything
torch.save(test_dataset, 'test_dataset.pt')

# --- Training ---
print("Initializing model and trainer...")
model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=len(id2label))

training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=16,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    output_dir='./results',
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='steps',
    report_to=[], 
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

print("Starting training...")
trainer.train()

# --- Save and Push ---
print("Pushing to Hugging Face Hub...")
trainer.push_to_hub(REPO_ID)
tokenizer.push_to_hub(REPO_ID)
print("Training complete and model pushed to Hub!")
