import json
import jiwer
import Levenshtein
import numpy as np
import torch

from hwr.ctc_decoder import build_ctc_decoder

# === Load vocab ===
with open("/home/mukul36/Documents/REWI/pd_wi_hw5_word/token_vocab.json", "r") as f:
    token_to_index = json.load(f)
    index_to_token = {int(v): k for k, v in token_to_index.items()}

# === Build list of categories (for decoder) ===
sorted_tokens = [token for idx, token in sorted(index_to_token.items())]
ctc_decoder = build_ctc_decoder(sorted_tokens, type="best_path")


# === Metrics ===
def get_levenshtein_distance(preds: list[str], labels: list[str]) -> tuple[float, float]:
    dist_leven = []
    len_label_avg = []

    for pred, label in zip(preds, labels):
        dist = Levenshtein.distance(pred, label)
        dist_leven.append(dist)
        len_label_avg.append(len(label))

    return np.mean(dist_leven), np.mean(len_label_avg)


def evaluate(
    preds: str | list[str],
    labels: str | list[str],
    use_ld: bool = True,
    use_cer: bool = True,
    use_wer: bool = True,
) -> dict:
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(labels, str):
        labels = [labels]

    if use_ld:
        dist_leven, len_sent_avg = get_levenshtein_distance(preds, labels)
    else:
        dist_leven, len_sent_avg = -1, -1

    cer = jiwer.cer(labels, preds) if use_cer else -1
    wer = jiwer.wer(labels, preds) if use_wer else -1

    return {
        'levenshtein_distance': dist_leven,
        'average_sentence_length': len_sent_avg,
        'character_error_rate': cer,
        'word_error_rate': wer,
    }


# === Prediction Evaluation ===
def evaluate_predictions(pred_index_seqs, label_index_seqs):
    decoded_preds = []
    decoded_labels = []

    for pred_idxs, label_idxs in zip(pred_index_seqs, label_index_seqs):
        # Ensure 1D tensors, even if single-element
        pred_tensor = torch.tensor(pred_idxs)
        label_tensor = torch.tensor(label_idxs)

        if pred_tensor.ndim == 0:
            pred_tensor = pred_tensor.unsqueeze(0)
        if label_tensor.ndim == 0:
            label_tensor = label_tensor.unsqueeze(0)

        pred_text = ctc_decoder.decode(pred_tensor, label=False)
        label_text = ctc_decoder.decode(label_tensor, label=True)

        decoded_preds.append(pred_text)
        decoded_labels.append(label_text)

        print(f"PRED:  {pred_text}")
        print(f"GT   :  {label_text}")
        print("---")

    return evaluate(decoded_preds, decoded_labels)
