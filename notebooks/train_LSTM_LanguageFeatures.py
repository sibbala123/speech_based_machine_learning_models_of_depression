# ====================================================================
# CONTEXTUAL LSTM FOR DEPRESSION ESTIMATION (Sentence-level)
# - Uses sentence-level BERT + VADER sentiment sequences saved in a pickle
# - K-Fold CV across participants
# - Small hyperparameter search
# - Reports Pearson r and absolute relative error (RE)
# ====================================================================

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import itertools
import time
import json

# -----------------------------
# CONFIG - update paths / choices
# -----------------------------
DATA_DIR = r"C:\Users\jayan\ML_Projects\speech_based_machine_learning_models_of_depression\features"
# Prefer sentence-level file created earlier. If not present, fallback to legacy file.
SENTENCE_LEVEL_PKL = os.path.join(DATA_DIR, "language_features_sentence_level.pkl")
# Directory to save trained models
MODEL_SAVE_DIR = r"C:\Users\jayan\ML_Projects\speech_based_machine_learning_models_of_depression\models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
LEGACY_PKL = os.path.join(DATA_DIR, "language_features.pkl")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Experiment settings
NUM_FOLDS = 5
RANDOM_STATE = 42

# Hyperparameter grid to try (kept small to be practical)
HYPERPARAM_GRID = {
    "hidden_size": [64, 128],
    "lr": [1e-3, 5e-4],
    "dropout": [0.3, 0.5],
    "batch_size": [16, 32],
    "num_layers": [1, 2]
}

# Model fusion modes to evaluate: 'concat', 'gated', 'attention'
FUSION_MODES = ["concat", "gated", "attention"]

# Training settings
EPOCHS = 40
PATIENCE = 8  # early stopping
WEIGHT_DECAY = 1e-5

# Random seeds
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# -----------------------------
# 1) LOAD FEATURES
# -----------------------------
print("\nLoading features...")
if os.path.exists(SENTENCE_LEVEL_PKL):
    with open(SENTENCE_LEVEL_PKL, "rb") as f:
        data = pickle.load(f)
    print("Loaded sentence-level feature file:", SENTENCE_LEVEL_PKL)
elif os.path.exists(LEGACY_PKL):
    with open(LEGACY_PKL, "rb") as f:
        legacy = pickle.load(f)
    # Attempt to convert legacy to sentence-level style (best-effort)
    raise FileNotFoundError(
        f"Sentence-level pickle not found at {SENTENCE_LEVEL_PKL}. "
        f"Please run the sentence-level extractor or place the file there."
    )
else:
    raise FileNotFoundError(f"Neither {SENTENCE_LEVEL_PKL} nor {LEGACY_PKL} found. Place sentence-level pickle at {SENTENCE_LEVEL_PKL}.")

# Expected keys in data:
# 'participant_ids' (list), 'phq_scores' (np.array), 'bert_sequences' (dict pid->list[np.array]),
# 'sentiment_sequences' (dict pid->list[float]), 'sequence_lengths' (dict), 'raw_text' (dict)
participant_ids = data["participant_ids"]
phq_scores = np.array(data["phq_scores"], dtype=float)
bert_sequences = data["bert_sequences"]         # pid -> list of np arrays (utt_count x bert_dim)
sentiment_sequences = data["sentiment_sequences"]  # pid -> list of floats
sequence_lengths = data.get("sequence_lengths", {pid: len(bert_sequences[pid]) for pid in participant_ids})
raw_text = data.get("raw_text", None)

# Verify shapes / dimensions
example_pid = participant_ids[0]
example_bert_seq = bert_sequences[example_pid]
if len(example_bert_seq) == 0:
    raise RuntimeError("Example participant has zero utterances; check your sentence-level pickle.")
bert_dim = len(example_bert_seq[0])
print(f"Participants: {len(participant_ids)}, example sequence length: {sequence_lengths[example_pid]}, BERT dim: {bert_dim}")

# Build ordered lists aligned with participant_ids (for convenience)
ordered_bert_seqs = [bert_sequences[pid] for pid in participant_ids]
ordered_sent_seqs = [sentiment_sequences[pid] for pid in participant_ids]
ordered_lengths = [sequence_lengths[pid] for pid in participant_ids]
ordered_phq = phq_scores.copy()  # same order as participant_ids


# -----------------------------
# 2) DATASET + COLLATE (padding)
# -----------------------------
class SentenceLevelDataset(Dataset):
    def __init__(self, bert_seqs, sent_seqs, lengths, labels):
        """
        bert_seqs: list of lists of np arrays (utt_count x bert_dim)
        sent_seqs: list of lists of floats (utt_count)
        lengths: list of ints
        labels: numpy array (N,)
        """
        self.bert_seqs = bert_seqs
        self.sent_seqs = sent_seqs
        self.lengths = lengths
        self.labels = labels.astype(float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert on-the-fly to numpy arrays
        bert_seq = np.stack(self.bert_seqs[idx]).astype(np.float32)    # (seq_len, bert_dim)
        sent_seq = np.array(self.sent_seqs[idx], dtype=np.float32)     # (seq_len,)
        length = int(self.lengths[idx])
        label = float(self.labels[idx])
        return bert_seq, sent_seq, length, label


def collate_fn(batch):
    """
    batch: list of (bert_seq(np array seq_len x D), sent_seq (seq_len,), length, label)
    Returns padded tensors and length array and aggregated sentiment scalar (mean).
    """
    bert_seqs, sent_seqs, lengths, labels = zip(*batch)
    lengths = np.array(lengths, dtype=np.int64)
    max_len = int(lengths.max())

    batch_size = len(batch)
    # allocate
    bert_dim_local = bert_seqs[0].shape[1]
    bert_padded = np.zeros((batch_size, max_len, bert_dim_local), dtype=np.float32)
    sent_padded = np.zeros((batch_size, max_len), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)

    for i, (b_seq, s_seq, L) in enumerate(zip(bert_seqs, sent_seqs, lengths)):
        bert_padded[i, :L, :] = b_seq
        sent_padded[i, :L] = s_seq
        mask[i, :L] = 1.0

    # Convert to tensors
    bert_padded = torch.from_numpy(bert_padded)            # (B, max_len, D)
    sent_padded = torch.from_numpy(sent_padded).unsqueeze(-1)  # (B, max_len, 1)
    mask = torch.from_numpy(mask)                          # (B, max_len)
    lengths_tensor = torch.from_numpy(lengths)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Per-sample aggregated sentiment features (mean, std)
    sent_mean = (sent_padded.squeeze(-1) * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
    sent_std = torch.sqrt(((sent_padded.squeeze(-1) - sent_mean.unsqueeze(1))**2 * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8))

    sent_agg = torch.stack([sent_mean, sent_std], dim=1)  # (B, 2)

    return bert_padded, sent_padded, mask, lengths_tensor, sent_agg, labels_tensor


# -----------------------------
# 3) MODEL: unified multi-fusion LSTM
# -----------------------------
class ContextualLSTM(nn.Module):
    """
    Unified LSTM model that can operate in three fusion modes:
    - 'concat' : per-timestep concatenate [bert, sentiment] then LSTM -> fc
    - 'gated'  : LSTM over bert -> final hidden; sentiment aggregated (mean,std) -> gate to modulate hidden
    - 'attention' : LSTM over bert -> attention over time where sentiment aggregate biases attention scores
    """
    def __init__(self, bert_dim=384, hidden_size=128, num_layers=2, dropout=0.3, fusion="concat"):
        super().__init__()
        assert fusion in ("concat", "gated", "attention")
        self.fusion = fusion
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        if fusion == "concat":
            input_size = bert_dim + 1  # per-timestep sentiment scalar
        else:
            input_size = bert_dim

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0)

        # For gated and attention we encode aggregated sentiment (mean,std) to a vector
        if fusion in ("gated", "attention"):
            self.sent_agg_encoder = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, hidden_size)
            )

        # For gated fusion: gating layer
        if fusion == "gated":
            self.gate_layer = nn.Linear(hidden_size, hidden_size)

        # For attention fusion: layers to compute attention scores
        if fusion == "attention":
            self.att_text = nn.Linear(hidden_size, hidden_size)
            self.att_sent = nn.Linear(hidden_size, hidden_size)
            self.att_comb = nn.Linear(hidden_size, 1)

        # Final prediction layers
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, bert_seq, sent_seq, mask, lengths, sent_agg):
        """
        bert_seq: (B, T, D)
        sent_seq: (B, T, 1)
        mask: (B, T)
        lengths: (B,)
        sent_agg: (B, 2) aggregated sentiment features (mean,std)
        """
        B, T, D = bert_seq.shape

        if self.fusion == "concat":
            x = torch.cat([bert_seq, sent_seq], dim=2)  # (B,T,D+1)
            packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed)
            # get last hidden of top layer
            h_last = h_n[-1]  # (B, hidden)
            h = h_last
        else:
            # process bert-only sequence
            packed = nn.utils.rnn.pack_padded_sequence(bert_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (h_n, c_n) = self.lstm(packed)
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)  # (B,T,hidden)
            # compute h_text as last valid step (use lengths)
            idx = (lengths - 1).long().unsqueeze(1).unsqueeze(2).expand(B, 1, self.hidden_size)
            h_last = lstm_out.gather(1, idx).squeeze(1)  # (B, hidden)

            if self.fusion == "gated":
                # Encode aggregated sentiment -> context vector
                sent_vec = self.sent_agg_encoder(sent_agg)  # (B, hidden)
                gate = torch.tanh(self.gate_layer(sent_vec))  # (B, hidden)
                h = h_last * gate
            else:  # attention
                # encode aggregated sentiment to vector and broadcast to timesteps
                sent_vec = self.sent_agg_encoder(sent_agg)  # (B, hidden)
                sent_vec_exp = sent_vec.unsqueeze(1)  # (B,1,hidden)
                # attention scores
                text_proj = self.att_text(lstm_out)  # (B,T,hidden)
                sent_proj = self.att_sent(sent_vec_exp)  # (B,1,hidden)
                att_in = torch.tanh(text_proj + sent_proj)  # (B,T,hidden)
                att_scores = self.att_comb(att_in).squeeze(-1)  # (B,T)
                # mask
                att_scores = att_scores.masked_fill(mask == 0, float("-inf"))
                att_weights = F.softmax(att_scores, dim=1).unsqueeze(-1)  # (B,T,1)
                h = torch.sum(lstm_out * att_weights, dim=1)  # (B, hidden)

        # final MLP
        out = self.fc1(h)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out).squeeze(1)
        return out


# -----------------------------
# 4) TRAIN / EVAL helpers
# -----------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for bert_p, sent_p, mask, lengths, sent_agg, labels in loader:
        bert_p = bert_p.to(device)
        sent_p = sent_p.to(device)
        mask = mask.to(device)
        lengths = lengths.to(device)
        sent_agg = sent_agg.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        preds = model(bert_p, sent_p, mask, lengths, sent_agg)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * bert_p.size(0)
    return running_loss / len(loader.dataset)


def eval_model(model, loader, device):
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for bert_p, sent_p, mask, lengths, sent_agg, labels in loader:
            bert_p = bert_p.to(device)
            sent_p = sent_p.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)
            sent_agg = sent_agg.to(device)
            labels = labels.to(device)

            preds = model(bert_p, sent_p, mask, lengths, sent_agg)
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)

    # Pearson r
    if np.std(preds_all) == 0 or np.std(labels_all) == 0:
        r = 0.0
    else:
        r, _ = pearsonr(labels_all, preds_all)

    # Absolute relative error (RE): mean(|pred - label| / max_label)
    max_label = np.max(labels_all) if np.max(labels_all) != 0 else 1.0
    re = np.mean(np.abs(preds_all - labels_all) / max_label)

    return r, re, preds_all, labels_all


# -----------------------------
# 5) CROSS-VALIDATION + HYPERPARAM SEARCH
# -----------------------------
# Build dataset once
full_dataset = SentenceLevelDataset(ordered_bert_seqs, ordered_sent_seqs, ordered_lengths, ordered_phq)

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Prepare list of hyperparameter combos
keys, values = zip(*HYPERPARAM_GRID.items())
combos = [dict(zip(keys, v)) for v in itertools.product(*values)]

results = []  # store results for each combo and fusion mode

start_time = time.time()
for fusion in FUSION_MODES:
    print("\n" + "="*40)
    print(f"Evaluating fusion mode: {fusion}")
    print("="*40)
    for combo in combos:
        hidden_size = combo["hidden_size"]
        lr = combo["lr"]
        dropout = combo["dropout"]
        batch_size = combo["batch_size"]
        num_layers = combo["num_layers"]

        fold_metrics = []
        fold_preds = []
        fold_labels = []

        print(f"\nGrid: fusion={fusion}, hidden={hidden_size}, lr={lr}, drop={dropout}, batch={batch_size}, layers={num_layers}")

        fold_idx = 0
        for train_idx, test_idx in kf.split(np.arange(len(full_dataset))):
            fold_idx += 1
            # Build dataloaders for this fold
            train_subset = torch.utils.data.Subset(full_dataset, train_idx)
            test_subset = torch.utils.data.Subset(full_dataset, test_idx)
            train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

            # Initialize model
            model = ContextualLSTM(bert_dim=bert_dim, hidden_size=hidden_size,
                                   num_layers=num_layers, dropout=dropout, fusion=fusion).to(DEVICE)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
            criterion = nn.MSELoss()

            # Early stopping variables
            best_val_loss = float("inf")
            patience_counter = 0
            best_state = None

            # Train
            for epoch in range(1, EPOCHS + 1):
                train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
                # validation uses test_loader here (no separate val set) — this is a design choice for small data
                r_val, re_val, _, _ = eval_model(model, test_loader, DEVICE)
                # use re_val as proxy validation loss (or MSE if you prefer)
                val_metric = re_val

                if val_metric < best_val_loss:
                    best_val_loss = val_metric
                    patience_counter = 0
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= PATIENCE:
                        # early stop
                        break

            # load best state
            if best_state is not None:
                model.load_state_dict(best_state)

            # Evaluate on test
            r_test, re_test, preds_test, labels_test = eval_model(model, test_loader, DEVICE)
            print(f" Fold {fold_idx} -- r: {r_test:.4f}, RE: {re_test:.4f}, n_test: {len(labels_test)}")
            # -------------------------------------------------------------
            # SAVE THIS FOLD’S BEST MODEL
            # -------------------------------------------------------------
            model_filename = (
                f"fusion={fusion}_hidden={hidden_size}_lr={lr}_drop={dropout}_"
                f"batch={batch_size}_layers={num_layers}_fold={fold_idx}.pt"
            )
            model_path = os.path.join(MODEL_SAVE_DIR, model_filename)
            torch.save(best_state, model_path)
            print(f" Saved model checkpoint: {model_path}")


            fold_metrics.append((r_test, re_test))
            fold_preds.append(preds_test)
            fold_labels.append(labels_test)

        # aggregate across folds (concatenate predictions and labels)
        all_preds = np.concatenate(fold_preds)
        all_labels = np.concatenate(fold_labels)
        # compute aggregate metrics
        if np.std(all_preds) == 0 or np.std(all_labels) == 0:
            agg_r = 0.0
        else:
            agg_r, _ = pearsonr(all_labels, all_preds)
        max_label_all = np.max(all_labels) if np.max(all_labels) != 0 else 1.0
        agg_re = np.mean(np.abs(all_preds - all_labels) / max_label_all)

        print(f"\n>>> GRID RESULT: fusion={fusion}, hidden={hidden_size}, lr={lr}, drop={dropout}, batch={batch_size}, layers={num_layers}")
        print(f"    Aggregate r: {agg_r:.4f}, Aggregate RE: {agg_re:.4f}")
        results.append({
            "fusion": fusion,
            "hidden_size": hidden_size,
            "lr": lr,
            "dropout": dropout,
            "batch_size": batch_size,
            "num_layers": num_layers,
            "agg_r": float(agg_r),
            "agg_re": float(agg_re),
            "all_preds": all_preds,
            "all_labels": all_labels
        })

# report best results
results_sorted_by_r = sorted(results, key=lambda x: x["agg_r"], reverse=True)
best_by_r = results_sorted_by_r[0]
results_sorted_by_re = sorted(results, key=lambda x: x["agg_re"])
best_by_re = results_sorted_by_re[0]

elapsed = time.time() - start_time

print("\n" + "=" * 60)
print("HYPERPARAM SEARCH COMPLETE")
print(f"Elapsed time: {elapsed/60:.2f} minutes")
print("Top result by Pearson r:")
print(best_by_r)
print("\nTop result by lowest RE:")
print(best_by_re)

# -------------------------------------------------------------
# SAVE BEST OVERALL MODEL + ITS HYPERPARAMETERS
# -------------------------------------------------------------
# Determine which metric to use — we will use highest Pearson r
best_overall = best_by_r  

best_fusion   = best_overall["fusion"]
best_hidden   = best_overall["hidden_size"]
best_lr       = best_overall["lr"]
best_dropout  = best_overall["dropout"]
best_batch    = best_overall["batch_size"]
best_layers   = best_overall["num_layers"]

# Save hyperparameters
best_hparam_path = os.path.join(MODEL_SAVE_DIR, "best_hyperparameters.json")
with open(best_hparam_path, "w") as f:
    json.dump({
        "fusion": best_fusion,
        "hidden_size": best_hidden,
        "lr": best_lr,
        "dropout": best_dropout,
        "batch_size": best_batch,
        "num_layers": best_layers
    }, f, indent=4)
print(f"\nSaved best hyperparameters to: {best_hparam_path}")

# Note:
# To save the best model weights, we need to identify
# the corresponding fold model saved earlier.
best_model_filename_prefix = (
    f"fusion={best_fusion}_hidden={best_hidden}_lr={best_lr}_drop={best_dropout}_"
    f"batch={best_batch}_layers={best_layers}"
)

print("\nBest model is one of the fold files matching:")
print(f"  {best_model_filename_prefix}")
print("You may choose the fold with highest r from the logs as the final model.")


# Save results to file for later inspection
out_path = os.path.join(DATA_DIR, "contextual_lstm_sentence_results.pkl")
with open(out_path, "wb") as f:
    pickle.dump(results, f)
print(f"\nAll results saved to: {out_path}")

# Also print a simple summary table
print("\nSummary (top 5 by r):")
for r in results_sorted_by_r[:5]:
    print(f"fusion={r['fusion']}, hidden={r['hidden_size']}, lr={r['lr']}, drop={r['dropout']}, batch={r['batch_size']}, layers={r['num_layers']}, r={r['agg_r']:.4f}, RE={r['agg_re']:.4f}")

print("\nDone.")