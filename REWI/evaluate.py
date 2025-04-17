import os
import torch
import yaml
import json
from torch.utils.data import DataLoader
from hwr.dataset import HRDataset
from hwr.model import get_model
from hwr.evaluate import evaluate_predictions
from torch.nn.utils.rnn import pad_sequence
from hwr.ctc_decoder import build_ctc_decoder  # ✅ Import decoder builder

# === Custom Collate Function ===
def custom_collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_seqs = pad_sequence(sequences, batch_first=True)
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    return padded_seqs, padded_labels

# === Load YAML Config ===
cfg_path = "/home/mukul36/Documents/REWI/configs/train_mukul.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# === Load Categories (tokens) ===
with open("/home/mukul36/Documents/REWI/pd_wi_hw5_word/token_vocab.json", "r") as f:
    token_to_index = json.load(f)
    index_to_token = {int(v): k for k, v in token_to_index.items()}

categories = [index_to_token[i] for i in range(len(index_to_token))]

# === Setup Device ===
device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load Evaluation Dataset ===
val_path = cfg["dir_dataset"].replace("train.json", "val.json")
val_dataset = HRDataset(
    path_anno=val_path,
    categories=cfg["categories"],
    sensors=cfg["sensors"],
    ratio_ds=cfg.get("ratio_ds", 8),
    idx_cv=cfg.get("idx_cv", 0),
    size_window=cfg.get("size_window", 1),
    aug=False,
    len_seq=cfg.get("len_seq", 0),
    cache=cfg.get("cache", False),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg["size_batch"],
    shuffle=False,
    num_workers=cfg.get("num_worker", 4),
    collate_fn=custom_collate_fn,
)

# === Load Model ===
checkpoint_path = os.path.join(cfg["dir_work"], "epoch_10.pth")
print(f"Loading model from: {checkpoint_path}")
model = get_model(
    arch_en=cfg["arch_en"],
    arch_de=cfg["arch_de"],
    in_chan=cfg["in_chan"],
    num_cls=cfg["num_cls"],
    ratio_ds=cfg.get("ratio_ds", 8),
    len_seq=cfg.get("len_seq", 0),
).to(device)

state = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state["model_state"])
model.eval()

# === Initialize CTC Decoder ===
decoder = build_ctc_decoder(categories, cfg.get("ctc_decoder", "best_path"))

# === Run Evaluation ===
decoded_preds, decoded_labels = [], []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        x = x.transpose(1, 2)  # [B, C, T]
        logits = model(x)      # [B, T, num_cls]

        for pred_seq, label_seq in zip(logits, y):
            pred_text = decoder.decode(pred_seq, label=False)
            label_text = decoder.decode(label_seq[label_seq != -1], label=True)

            decoded_preds.append(pred_text)
            decoded_labels.append(label_text)

            print(f"PRED: {pred_text}")
            print(f"GT  : {label_text}")
            print("---")

# === Evaluate
results = evaluate_predictions(decoded_preds, decoded_labels)

print("\n✅ Final Evaluation on val.json:")
for k, v in results.items():
    print(f"{k}: {v:.4f}")
