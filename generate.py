# Imports
import sys
import torch
from tqdm import tqdm
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Get command-line arguments
model_name = sys.argv[1]
tokenizer_name = sys.argv[2]
device = sys.argv[3]
doc_ids = sys.argv[4:]

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
tokenizer = PegasusTokenizer.from_pretrained(tokenizer_name)
print("Created tokenizer.")
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
print("Created model.")

# Iterate over documents
print("Creating summaries...")
for doc_id in doc_ids:

    # Print the actual summary
    print(f"\n{doc_id} -- ACTUAL\n")
    print(abstracts[doc_id])

    # Print header for generated summary
    print(f"\n{doc_id} -- GENERATED\n")

    # Tokenize and put through model
    batch = tokenizer(documents[doc_id], truncation = True, padding = "longest", return_tensors = "pt").to(device)
    translated = model.generate(**batch)
    summary = tokenizer.batch_decode(translated, skip_special_tokens = True)

    # Print generated summary
    print(summary[0])
