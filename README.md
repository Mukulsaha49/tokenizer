# tokenizer# REWI Token-Based Handwriting Recognition

This project implements a handwriting recognition system using sensor data from a digital pen. It uses a CNN-LSTM architecture trained with CTC loss, and evaluates the predictions using token-based metrics such as CER (Character Error Rate), WER (Word Error Rate), and Levenshtein distance. Tokenization is done using bigrams.

---

## 📁 Folder Structure

```
REWI/
├── hwr/
│   ├── dataset/              # Dataset loading and preprocessing
│   ├── model/                # Model architectures (conv.py and lstm.py)
│   ├── evaluate.py           # Evaluation logic (CTC decoding + metric calculation)
│   ├── ctc_decoder.py        # Best path CTC decoder
│   ├── tokenizer.py          # Tokenizer (bigrams)
│   └── ...
├── configs/
│   └── train_mukul.yaml      # Training and evaluation configuration
├── pd_wi_hw5_word/
│   ├── train.json            # Training dataset
│   ├── val.json              # Validation dataset
│   ├── annotation.json       # Complete annotation
│   └── token_vocab.json      # Generated token vocabulary
├── generate_token_vocab.py   # Script to build bigram token vocabulary
├── main_mukul.py             # Training script
├── evaluate.py               # Final evaluation script using val.json
└── README.md
```

---

## 🧠 Model

- **Encoder**: `conv.py` → `BLCNN` — 1D Convolution layers
- **Decoder**: `lstm.py` → `LSTM` — Bi-directional LSTM with Softmax
- **Training Loss**: CTC (Connectionist Temporal Classification)
- **CTC Decoder**: `best_path` decoding from `ctc_decoder.py`

---

## ✅ Step-by-Step Setup and Execution

### 1️⃣ Install Requirements

```bash
conda create -n ml_env python=3.10 -y
conda activate ml_env
pip install torch torchvision torchaudio
pip install numpy pyyaml jiwer python-Levenshtein
```

---

### 2️⃣ Generate Token Vocabulary

This step creates `token_vocab.json` with bigram tokens.

```bash
python generate_token_vocab.py
```

---

### 3️⃣ Verify YAML Configuration

Make sure your `train_mukul.yaml` looks like this:

```yaml
arch_en: blcnn
arch_de: lstm
aug: false
cache: true
checkpoint: null
ctc_decoder: best_path
device: cuda
dir_dataset: /path/to/pd_wi_hw5_word/train.json
dir_work: /path/to/pd_wi_hw5_word
epoch: 10
freq_eval: 5
freq_log: 10
freq_save: 5
idx_cv: 0
in_chan: 13
lr: 0.001
num_cls: 1261
num_worker: 4
seed: 42
sensors: [AF, AR, G, M, F]
size_batch: 64
test: false
categories: ["token"]
```

---

### 4️⃣ Start Training

```bash
python main_mukul.py
```

- Checkpoints will be saved under `dir_work/epoch_*.pth`
- Evaluation on `train.json` happens every `freq_eval` epochs

---

### 5️⃣ Final Evaluation using val.json

```bash
python evaluate.py
```

This script:
- Loads the trained model
- Runs inference on `val.json`
- Uses `best_path` decoding
- Prints and saves CER, WER, and Levenshtein distance

---

## 📊 Evaluation Metrics

- **CER (Character Error Rate)**: Percentage of incorrect characters
- **WER (Word Error Rate)**: Percentage of incorrect words
- **Levenshtein Distance**: Total edit distance between predicted and true sentence

---

## 💡 Notes

- Uses custom `collate_fn` to handle variable-length sequences
- Tokens are bigrams: `["he", "el", "ll", "lo"]` for `"hello"`
- `conv.py` and `lstm.py` are now the only model components in use (FEL models fully removed)

---

## 🧾 Credits

- Base project: REWI repository (RWTH Aachen University)
- Extended by: Mukul for token-based handwriting recognition
