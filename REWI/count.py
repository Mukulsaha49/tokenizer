import json
with open("/home/mukul36/Documents/REWI/Converted data/token_vocab.json", "r") as f:
    vocab = json.load(f)
print(len(vocab))  # let's say it prints 42
