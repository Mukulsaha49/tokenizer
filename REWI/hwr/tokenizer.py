import json
from collections import Counter

# Define special tokens
BLANK_TOKEN = "<BLANK>"  # CTC requires this at index 0
UNK_TOKEN = "<UNK>"


def tokenize(text, n=2):
    """Tokenize text into n-grams (e.g., bigrams)."""
    text = text.strip()
    return [text[i:i+n] for i in range(len(text) - n + 1)]


def build_vocab(texts, n=2, max_vocab_size=None):
    token_counter = Counter()

    for text in texts:
        tokens = tokenize(text, n)
        token_counter.update(tokens)

    # Special tokens go first: CTC needs <BLANK> at index 0, then <UNK>
    vocab = [BLANK_TOKEN, UNK_TOKEN]

    for token, _ in token_counter.most_common(max_vocab_size):
        if token not in vocab:
            vocab.append(token)

    token_to_index = {token: idx for idx, token in enumerate(vocab)}
    index_to_token = {idx: token for token, idx in token_to_index.items()}

    return vocab, token_to_index, index_to_token


def encode(text, token_to_index, n=2):
    tokens = tokenize(text, n)
    return [token_to_index.get(token, token_to_index[UNK_TOKEN]) for token in tokens]


def decode(indices, index_to_token):
    return [index_to_token.get(idx, UNK_TOKEN) for idx in indices]


def save_vocab(token_to_index, filepath):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(token_to_index, f, ensure_ascii=False, indent=4)


def load_vocab(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        token_to_index = json.load(f)
    index_to_token = {int(idx): token for idx, token in token_to_index.items()}
    return token_to_index, index_to_token
