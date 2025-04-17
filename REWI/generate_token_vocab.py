import json
from hwr.tokenizer import build_vocab, save_vocab

train_dataset = "/home/mukul36/Documents/REWI/pd_wi_hw5_word/train.json"  # Update if needed

# Load annotations
with open(train_dataset, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract labels from fold "0"
annotations = data["annotations"]["0"]
texts = [entry["label"] for entry in annotations]

# Build bigram vocab
_, token_to_index, _ = build_vocab(texts, n=2)

# === Add special tokens (force "<BLANK>" at index 0, "<UNK>" at index 1) ===
# Remove if they exist to avoid duplicates
token_to_index.pop("<PAD>", None)
token_to_index.pop("<UNK>", None)
token_to_index.pop("<BLANK>", None)

# Rebuild vocab with required order
ordered_tokens = ["<BLANK>", "<UNK>"] + list(token_to_index.keys())
token_to_index = {token: idx for idx, token in enumerate(ordered_tokens)}

# Save vocab
output_path = "/home/mukul36/Documents/REWI/pd_wi_hw5_word/token_vocab.json"
save_vocab(token_to_index, output_path)

print(f"✅ Vocabulary saved to: {output_path}")
print(f"🧠 Vocab size: {len(token_to_index)} tokens")
