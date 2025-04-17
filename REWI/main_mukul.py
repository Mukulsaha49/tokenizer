import os
import torch
import yaml
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from hwr.dataset import HRDataset
from hwr.model import get_model
from hwr.evaluate import evaluate_predictions

# === Custom Collate Function ===
def custom_collate_fn(batch):
    sequences, labels = zip(*batch)

    padded_seqs = pad_sequence(sequences, batch_first=True)  # [B, T, C]
    padded_labels = pad_sequence(labels, batch_first=True, padding_value=-1)  # [B, T]

    return padded_seqs, padded_labels

# === Load Config ===
cfg_path = "/home/mukul36/Documents/REWI/configs/train_mukul.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)

# === Device ===
device = torch.device(cfg["device"] if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === Load Dataset ===
print("Loading dataset...")
train_dataset = HRDataset(
    path_anno=cfg["dir_dataset"],
    categories=cfg["categories"],
    sensors=cfg["sensors"],
    ratio_ds=cfg.get("ratio_ds", 8),
    idx_cv=str(cfg.get("idx_cv", 0)),
    size_window=cfg.get("size_window", 1),
    aug=cfg.get("aug", False),
    len_seq=cfg.get("len_seq", 0),
    cache=cfg.get("cache", False),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg["size_batch"],
    shuffle=True,
    num_workers=cfg.get("num_worker", 4),
    collate_fn=custom_collate_fn,
)

# === Build Model ===
print("Building model...")
model = get_model(
    arch_en=cfg["arch_en"],
    arch_de=cfg["arch_de"],
    in_chan=cfg["in_chan"],
    num_cls=cfg["num_cls"],
    ratio_ds=cfg.get("ratio_ds", 8),
    len_seq=cfg.get("len_seq", 0),
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
ctc_loss = torch.nn.CTCLoss(blank=0, zero_infinity=True)

# === Training Loop ===
print("Starting training...")
model.train()
for epoch in range(1, cfg["epoch"] + 1):
    total_loss = 0.0

    for i, (x, y) in enumerate(train_loader):
        x = x.to(device)
        y = y.to(device)

        x = x.permute(0, 2, 1)  # [B, C, T]

        label_lengths = (y != -1).sum(dim=1)
        y = y[y != -1]  # flatten

        optimizer.zero_grad()
        logits = model(x)  # [B, T, num_cls]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        input_lengths = torch.full((x.size(0),), logits.size(1), dtype=torch.long)

        loss = ctc_loss(
            log_probs.transpose(0, 1),
            y,
            input_lengths,
            label_lengths,
        )

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if (i + 1) % cfg["freq_log"] == 0:
            print(f"Epoch [{epoch}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"\n✅ Epoch [{epoch}] completed | Avg Loss: {avg_loss:.4f}")

    # === Evaluation ===
    if epoch % cfg["freq_eval"] == 0:
        print("Evaluating on training data...")
        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(device)
                y = y.to(device)

                x = x.permute(0, 2, 1)  # [B, C, T]
                logits = model(x)
                preds = torch.argmax(logits, dim=-1)

                # ✅ Prevent flattening — maintain [B, T] structure
                all_preds.extend(preds.cpu().numpy().tolist())
                all_labels.extend(y.cpu().numpy().tolist())

        results = evaluate_predictions(all_preds, all_labels)
        print("\n📊 Evaluation Results:")
        for k, v in results.items():
            print(f"{k}: {v:.4f}")
        model.train()

    # === Save Checkpoint ===
    if epoch % cfg["freq_save"] == 0:
        save_dir = cfg.get("dir_work", "outputs")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
        torch.save({"model_state": model.state_dict()}, save_path)
        print(f"💾 Model checkpoint saved at {save_path}\n")
