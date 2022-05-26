# Imports
import sys
import pandas as pd
import torch
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
from datasets import load_metric

# Get the suffix and device
suffix = sys.argv[1]
device = sys.argv[2]

# Get the list of document IDs
with open("scisummnet-parsed/valid-ids.txt", "r") as f:
    doc_ids = f.read().splitlines()

# Subset for script testing
# doc_ids = doc_ids[:25]

# Load in all the documents and abstracts
print("Loading in documents and abstracts...")
documents = {}
abstracts = {}
for doc_id in tqdm(doc_ids, ncols = 64):
    root = f"scisummnet-parsed/{doc_id}"
    with open(f"{root}/abstract.txt", "r") as f:
        abstracts[doc_id] = f.read()
    with open(f"{root}/full.txt", "r") as f:
        documents[doc_id] = " ".join(f.read().splitlines())

# Perform the split
train_frac = 0.85
train_num = int(len(doc_ids) * train_frac)
train_docs = doc_ids[:train_num]
val_docs = doc_ids[train_num:]
train_texts, train_labels = [documents[doc_id] for doc_id in train_docs], [abstracts[doc_id] for doc_id in train_docs]
val_texts, val_labels = [documents[doc_id] for doc_id in val_docs], [abstracts[doc_id] for doc_id in val_docs]
print(f"Used fraction of {train_frac} to split into {len(train_docs)} for training and {len(val_docs)} for testing.")

# Define dataset class
class PegasusDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels['input_ids'][idx])
        return item
    def __len__(self):
        return len(self.labels['input_ids'])

# Define data preparation function
def prepare_data(model_name, train_texts, train_labels, val_texts, val_labels):

    # Define the tokenizer
    tokenizer = PegasusTokenizer.from_pretrained(model_name)

    # Sub-function to tokenize given data
    def tokenize_data(texts, labels):
        encodings = tokenizer(texts, truncation = True, padding = "longest")
        decodings = tokenizer(labels, truncation = True, padding = "longest")
        dataset_tokenized = PegasusDataset(encodings, decodings)
        return dataset_tokenized

    # Perform the tokenization
    train_dataset = tokenize_data(train_texts, train_labels)
    val_dataset = tokenize_data(val_texts, val_labels)

    return train_dataset, val_dataset, tokenizer

# Prepare the dataset
model_name = f"google/pegasus-{suffix}"
train_dataset, val_dataset, tokenizer = prepare_data(model_name, train_texts, train_labels, val_texts, val_labels)
print("Prepared data.")

# Load the model
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
print("Loaded model.")

# Set up training arguments
training_args = TrainingArguments(
    output_dir = f"fine-tune/{suffix}",
    save_total_limit = 10,
    num_train_epochs = 10,
    per_device_train_batch_size = 1,
    per_device_eval_batch_size = 1,
    evaluation_strategy = "epoch",
    weight_decay = 0.01,
)

# Set up trainer
trainer = Trainer(
    model = model,
    args = training_args,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    tokenizer = tokenizer,
)

# Train!
trainer.train()
