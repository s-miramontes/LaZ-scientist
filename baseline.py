# Compute baseline ROUGE scores for summaries from non-fine-tuned
# PEGASUS models, and save the results to file.

# USAGE: python baseline.py [suffix]

# EXAMPLE: python baseline.py arxiv
# This will use the google/pegasus-arxiv model and
# save the output to baselines/arxiv.csv.

# Imports
import sys
import pandas as pd
import torch
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from datasets import load_metric

# Get the supplied suffix
suffix = sys.argv[1]

# Get the list of document IDs
with open("scisummnet-parsed/valid-ids.txt", "r") as f:
    doc_ids = f.read().splitlines()

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

# Create processing objects
model_name = f"google/pegasus-{suffix}"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Selected device: {device}.")
tokenizer = PegasusTokenizer.from_pretrained(model_name)
print("Created tokenizer.")
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
print("Created model.")
metric = load_metric("rouge")
print("Created metric.")

# Iterate over documents
print("Summarizing documents and recording scores...")
scores = {"doc_id": [], "rouge1": [], "rouge2": [], "rougeL": []}
for doc_id in tqdm(doc_ids, ncols = 64):

    # Add document ID to dictionary (redundant but useful for dataframe)
    scores["doc_id"].append(doc_id)

    # Tokenize and put through model
    batch = tokenizer(documents[doc_id], truncation = True, padding = "longest", return_tensors = "pt").to(device)
    translated = model.generate(**batch)
    summary = tokenizer.batch_decode(translated, skip_special_tokens = True)

    # Calculate and record scores
    score = metric.compute(predictions = summary, references = [abstracts[doc_id]])
    scores["rouge1"].append(score["rouge1"].mid.fmeasure)
    scores["rouge2"].append(score["rouge2"].mid.fmeasure)
    scores["rougeL"].append(score["rougeL"].mid.fmeasure)

# Save results
outfile = f"baselines/{suffix}.csv"
scores_df = pd.DataFrame.from_dict(scores)
scores_df.to_csv(outfile, index = False)
print(f"Saved results to {outfile}.")