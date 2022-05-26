# Imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Get the list of document IDs
with open("scisummnet-parsed/valid-ids.txt", "r") as f:
    doc_ids = f.read().splitlines()

# Split into training and validation
train_ids = doc_ids[:778]
val_ids = doc_ids[778:]

# Load in scoring data
print("Loading raw data...")
before_arxiv = pd.read_csv("scores/baselines/arxiv.csv")
before_pubmed = pd.read_csv("scores/baselines/pubmed.csv")
after_arxiv = pd.read_csv("scores/checkpoint-3500/arxiv.csv")
after_pubmed = pd.read_csv("scores/checkpoint-3500/pubmed.csv")

# Initialize score-based dataframes
column_names = ["doc_id", "set", "model", "status", "score"]
rouge1 = pd.DataFrame(columns = column_names)
rouge2 = pd.DataFrame(columns = column_names)
rougeL = pd.DataFrame(columns = column_names)

# Populate dataframes with training document data
print("Building dataframes with training documents...")
for doc_id in train_ids:
    rouge1 = rouge1.append({"doc_id": doc_id, "set": "train", "model": "arxiv", "status": "untuned", "score": float(before_arxiv[before_arxiv["doc_id"] == doc_id]["rouge1"])}, ignore_index = True)
    rouge2 = rouge2.append({"doc_id": doc_id, "set": "train", "model": "arxiv", "status": "untuned", "score": float(before_arxiv[before_arxiv["doc_id"] == doc_id]["rouge2"])}, ignore_index = True)
    rougeL = rougeL.append({"doc_id": doc_id, "set": "train", "model": "arxiv", "status": "untuned", "score": float(before_arxiv[before_arxiv["doc_id"] == doc_id]["rougeL"])}, ignore_index = True)
    rouge1 = rouge1.append({"doc_id": doc_id, "set": "train", "model": "arxiv", "status": "tuned", "score": float(after_arxiv[after_arxiv["doc_id"] == doc_id]["rouge1"])}, ignore_index = True)
    rouge2 = rouge2.append({"doc_id": doc_id, "set": "train", "model": "arxiv", "status": "tuned", "score": float(after_arxiv[after_arxiv["doc_id"] == doc_id]["rouge2"])}, ignore_index = True)
    rougeL = rougeL.append({"doc_id": doc_id, "set": "train", "model": "arxiv", "status": "tuned", "score": float(after_arxiv[after_arxiv["doc_id"] == doc_id]["rougeL"])}, ignore_index = True)
    rouge1 = rouge1.append({"doc_id": doc_id, "set": "train", "model": "pubmed", "status": "untuned", "score": float(before_pubmed[before_pubmed["doc_id"] == doc_id]["rouge1"])}, ignore_index = True)
    rouge2 = rouge2.append({"doc_id": doc_id, "set": "train", "model": "pubmed", "status": "untuned", "score": float(before_pubmed[before_pubmed["doc_id"] == doc_id]["rouge2"])}, ignore_index = True)
    rougeL = rougeL.append({"doc_id": doc_id, "set": "train", "model": "pubmed", "status": "untuned", "score": float(before_pubmed[before_pubmed["doc_id"] == doc_id]["rougeL"])}, ignore_index = True)
    rouge1 = rouge1.append({"doc_id": doc_id, "set": "train", "model": "pubmed", "status": "tuned", "score": float(after_pubmed[after_pubmed["doc_id"] == doc_id]["rouge1"])}, ignore_index = True)
    rouge2 = rouge2.append({"doc_id": doc_id, "set": "train", "model": "pubmed", "status": "tuned", "score": float(after_pubmed[after_pubmed["doc_id"] == doc_id]["rouge2"])}, ignore_index = True)
    rougeL = rougeL.append({"doc_id": doc_id, "set": "train", "model": "pubmed", "status": "tuned", "score": float(after_pubmed[after_pubmed["doc_id"] == doc_id]["rougeL"])}, ignore_index = True)

# Populate dataframes with validation document data
print("Building dataframes with validation documents...")
for doc_id in val_ids:
    rouge1 = rouge1.append({"doc_id": doc_id, "set": "val", "model": "arxiv", "status": "untuned", "score": float(before_arxiv[before_arxiv["doc_id"] == doc_id]["rouge1"])}, ignore_index = True)
    rouge2 = rouge2.append({"doc_id": doc_id, "set": "val", "model": "arxiv", "status": "untuned", "score": float(before_arxiv[before_arxiv["doc_id"] == doc_id]["rouge2"])}, ignore_index = True)
    rougeL = rougeL.append({"doc_id": doc_id, "set": "val", "model": "arxiv", "status": "untuned", "score": float(before_arxiv[before_arxiv["doc_id"] == doc_id]["rougeL"])}, ignore_index = True)
    rouge1 = rouge1.append({"doc_id": doc_id, "set": "val", "model": "arxiv", "status": "tuned", "score": float(after_arxiv[after_arxiv["doc_id"] == doc_id]["rouge1"])}, ignore_index = True)
    rouge2 = rouge2.append({"doc_id": doc_id, "set": "val", "model": "arxiv", "status": "tuned", "score": float(after_arxiv[after_arxiv["doc_id"] == doc_id]["rouge2"])}, ignore_index = True)
    rougeL = rougeL.append({"doc_id": doc_id, "set": "val", "model": "arxiv", "status": "tuned", "score": float(after_arxiv[after_arxiv["doc_id"] == doc_id]["rougeL"])}, ignore_index = True)
    rouge1 = rouge1.append({"doc_id": doc_id, "set": "val", "model": "pubmed", "status": "untuned", "score": float(before_pubmed[before_pubmed["doc_id"] == doc_id]["rouge1"])}, ignore_index = True)
    rouge2 = rouge2.append({"doc_id": doc_id, "set": "val", "model": "pubmed", "status": "untuned", "score": float(before_pubmed[before_pubmed["doc_id"] == doc_id]["rouge2"])}, ignore_index = True)
    rougeL = rougeL.append({"doc_id": doc_id, "set": "val", "model": "pubmed", "status": "untuned", "score": float(before_pubmed[before_pubmed["doc_id"] == doc_id]["rougeL"])}, ignore_index = True)
    rouge1 = rouge1.append({"doc_id": doc_id, "set": "val", "model": "pubmed", "status": "tuned", "score": float(after_pubmed[after_pubmed["doc_id"] == doc_id]["rouge1"])}, ignore_index = True)
    rouge2 = rouge2.append({"doc_id": doc_id, "set": "val", "model": "pubmed", "status": "tuned", "score": float(after_pubmed[after_pubmed["doc_id"] == doc_id]["rouge2"])}, ignore_index = True)
    rougeL = rougeL.append({"doc_id": doc_id, "set": "val", "model": "pubmed", "status": "tuned", "score": float(after_pubmed[after_pubmed["doc_id"] == doc_id]["rougeL"])}, ignore_index = True)

# Create plots
print("Saving plots...")
sns.violinplot(x = "model", y = "score", hue = "status", data = rouge1)
plt.title("ROUGE-1 (all)")
plt.tight_layout()
plt.savefig("plots/all-rouge1.png")
plt.clf()

sns.violinplot(x = "model", y = "score", hue = "status", data = rouge2)
plt.title("ROUGE-2 (all)")
plt.tight_layout()
plt.savefig("plots/all-rouge2.png")
plt.clf()

sns.violinplot(x = "model", y = "score", hue = "status", data = rougeL)
plt.title("ROUGE-L (all)")
plt.tight_layout()
plt.savefig("plots/all-rougeL.png")
plt.clf()

sns.violinplot(x = "model", y = "score", hue = "status", data = rouge1[rouge1["set"] == "val"])
plt.title("ROUGE-1 (val)")
plt.tight_layout()
plt.savefig("plots/val-rouge1.png")
plt.clf()

sns.violinplot(x = "model", y = "score", hue = "status", data = rouge2[rouge2["set"] == "val"])
plt.title("ROUGE-2 (val)")
plt.tight_layout()
plt.savefig("plots/val-rouge2.png")
plt.clf()

sns.violinplot(x = "model", y = "score", hue = "status", data = rougeL[rougeL["set"] == "val"])
plt.title("ROUGE-L (val)")
plt.tight_layout()
plt.savefig("plots/val-rougeL.png")
plt.clf()