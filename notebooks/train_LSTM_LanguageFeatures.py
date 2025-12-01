import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from scipy.stats import pearsonr
from collections import defaultdict
import itertools
import time
import json
import sys

# ====================================================================
# CONFIGURATION
# ====================================================================
DATA_DIR = r"C:\Users\jayan\ML_Projects\speech_based_machine_learning_models_of_depression\features"
SENTENCE_LEVEL_PKL = os.path.join(DATA_DIR, "language_features_sentence_level.pkl")
MODEL_SAVE_DIR = r"C:\Users\jayan\ML_Projects\speech_based_machine_learning_models_of_depression\models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
LEGACY_PKL = os.path.join(DATA_DIR, "language_features.pkl")

# --- GPU/Device Check ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)
if DEVICE.type == 'cpu':
    print("WARNING: CUDA not available. Training will be slow.")

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

# ====================================================================
# 1) DATA LOADING AND UTILITIES
# ====================================================================
def load_features():
    """Loads and preprocesses feature data."""
    print("\nLoading features...")
    if not os.path.exists(SENTENCE_LEVEL_PKL):
        raise FileNotFoundError(
            f"Sentence-level pickle not found at {SENTENCE_LEVEL_PKL}. "
            f"Please run the sentence-level extractor or place the file there."
        )

    with open(SENTENCE_LEVEL_PKL, "rb") as f:
        data = pickle.load(f)
    print("Loaded sentence-level feature file:", SENTENCE_LEVEL_PKL)

    participant_ids = data["participant_ids"]
    phq_scores = np.array(data["phq_scores"], dtype=float)
    bert_sequences = data["bert_sequences"]
    sentiment_sequences = data["sentiment_sequences"]
    sequence_lengths = data.get("sequence_lengths", {pid: len(bert_sequences[pid]) for pid in participant_ids})

    example_pid = participant_ids[0]
    example_bert_seq = bert_sequences[example_pid]
    if len(example_bert_seq) == 0:
        raise RuntimeError("Example participant has zero utterances; check your sentence-level pickle.")
    bert_dim = len(example_bert_seq[0])
   
    print(f"Participants: {len(participant_ids)}, BERT dim: {bert_dim}")

    ordered_bert_seqs = [bert_sequences[pid] for pid in participant_ids]
    ordered_sent_seqs = [sentiment_sequences[pid] for pid in participant_ids]
    ordered_lengths = [sequence_lengths[pid] for pid in participant_ids]
    ordered_phq = phq_scores.copy()

    return ordered_bert_seqs, ordered_sent_seqs, ordered_lengths, ordered_phq, bert_dim


class SentenceLevelDataset(Dataset):
    """Dataset for sentence-level features."""
    def __init__(self, bert_seqs, sent_seqs, lengths, labels):
        self.bert_seqs = bert_seqs
        self.sent_seqs = sent_seqs
        self.lengths = lengths
        self.labels = labels.astype(float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        bert_seq = np.stack(self.bert_seqs[idx]).astype(np.float32)
        sent_seq = np.array(self.sent_seqs[idx], dtype=np.float32)
        length = int(self.lengths[idx])
        label = float(self.labels[idx])
        return bert_seq, sent_seq, length, label


def collate_fn(batch):
    """Pads sequences and computes aggregated sentiment features."""
    bert_seqs, sent_seqs, lengths, labels = zip(*batch)
    lengths = np.array(lengths, dtype=np.int64)
    max_len = int(lengths.max())

    batch_size = len(batch)
    bert_dim_local = bert_seqs[0].shape[1]
   
    bert_padded = np.zeros((batch_size, max_len, bert_dim_local), dtype=np.float32)
    sent_padded = np.zeros((batch_size, max_len), dtype=np.float32)
    mask = np.zeros((batch_size, max_len), dtype=np.float32)

    for i, (b_seq, s_seq, L) in enumerate(zip(bert_seqs, sent_seqs, lengths)):
        bert_padded[i, :L, :] = b_seq
        sent_padded[i, :L] = s_seq
        mask[i, :L] = 1.0

    bert_padded = torch.from_numpy(bert_padded)
    sent_padded = torch.from_numpy(sent_padded).unsqueeze(-1)
    mask = torch.from_numpy(mask)
    lengths_tensor = torch.from_numpy(lengths)
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Per-sample aggregated sentiment features (mean, std)
    # Ensure mean calculation is stable
    mask_sum = mask.sum(dim=1)
    sent_mean = (sent_padded.squeeze(-1) * mask).sum(dim=1) / (mask_sum + 1e-8)
   
    # Calculate standard deviation
    squared_diff = (sent_padded.squeeze(-1) - sent_mean.unsqueeze(1))**2 * mask
    sent_std = torch.sqrt(squared_diff.sum(dim=1) / (mask_sum + 1e-8))

    sent_agg = torch.stack([sent_mean, sent_std], dim=1)  # (B, 2)

    return bert_padded, sent_padded, mask, lengths_tensor, sent_agg, labels_tensor

# ====================================================================
# 2) MODEL ARCHITECTURE
# ====================================================================
class ContextualLSTM(nn.Module):
    """
    Unified LSTM model with different fusion mechanisms for BERT and VADER sentiment.
    """
    def __init__(self, bert_dim=384, hidden_size=128, num_layers=2, dropout=0.3, fusion="concat", bidirectional=True):
        super().__init__()
        assert fusion in ("concat", "gated", "attention")
        self.fusion = fusion
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        if fusion == "concat":
            input_size = bert_dim + 1  # per-timestep sentiment scalar
        else:
            input_size = bert_dim

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0,
                            bidirectional=bidirectional)
        
        # Layer Normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)

        if fusion in ("gated", "attention"):
            self.sent_agg_encoder = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, hidden_size * self.num_directions)
            )

        if fusion == "gated":
            self.gate_layer = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)

        if fusion == "attention":
            self.att_text = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)
            self.att_sent = nn.Linear(hidden_size * self.num_directions, hidden_size * self.num_directions)
            self.att_comb = nn.Linear(hidden_size * self.num_directions, 1)

        self.fc1 = nn.Linear(hidden_size * self.num_directions, 64)
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
       
        # --- Common LSTM operation for all modes ---
        if self.fusion == "concat":
            x = torch.cat([bert_seq, sent_seq], dim=2)  # (B,T,D+1)
        else:
            x = bert_seq
       
        # Ensure lengths are on CPU for packing
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        
        # Unpack
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)  # (B,T,hidden*num_directions)
        
        # Apply LayerNorm
        lstm_out = self.layer_norm(lstm_out)

        if self.fusion == "concat":
            # Last hidden state logic for bidirectional
            # We can't just take h_n[-1] because it might only be one direction.
            # Instead, we gather the output at the last valid timestep for each sequence.
            
            # idx: (B, 1, hidden_dim)
            idx = (lengths - 1).long().view(B, 1, 1).expand(B, 1, self.hidden_size * self.num_directions)
            h = lstm_out.gather(1, idx).squeeze(1) # (B, hidden*num_directions)

        else:
            # For gated/attention, we use the full sequence or last state
           
            # Use last hidden state for h_last (for gated fusion)
            idx = (lengths - 1).long().view(B, 1, 1).expand(B, 1, self.hidden_size * self.num_directions)
            h_last = lstm_out.gather(1, idx).squeeze(1)  # (B, hidden*num_directions)
           
            if self.fusion == "gated":
                # Gated Fusion: h_last * Gate(sent_agg)
                sent_vec = self.sent_agg_encoder(sent_agg)  # (B, hidden*num_directions)
                gate = torch.sigmoid(self.gate_layer(sent_vec))  # Use sigmoid for a soft gate
                h = h_last * gate
            else:  # attention
                # Attention Fusion: Context Vector = sum(LSTM_out * Att_Weights)
                sent_vec = self.sent_agg_encoder(sent_agg)  # (B, hidden*num_directions)
                sent_vec_exp = sent_vec.unsqueeze(1)  # (B,1,hidden*num_directions)
               
                # Attention scores
                text_proj = self.att_text(lstm_out)
                sent_proj = self.att_sent(sent_vec_exp)
                att_in = torch.tanh(text_proj + sent_proj)
                att_scores = self.att_comb(att_in).squeeze(-1)  # (B,T)
               
                # Mask and softmax
                att_scores = att_scores.masked_fill(mask == 0, float("-inf"))
                att_weights = F.softmax(att_scores, dim=1).unsqueeze(-1)  # (B,T,1)
                h = torch.sum(lstm_out * att_weights, dim=1)  # (B, hidden*num_directions)

        # final MLP
        out = self.fc1(h)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out).squeeze(1)
        return out

# ====================================================================
# 3) TRAIN / EVAL HELPER FUNCTIONS
# ====================================================================
def train_epoch(model, loader, optimizer, criterion, device):
    """Performs one training epoch."""
    model.train()
    running_loss = 0.0
    num_samples = 0
    for bert_p, sent_p, mask, lengths, sent_agg, labels in loader:
        # Move all necessary tensors to device
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
       
        # --- Gradient Clipping to prevent explosion (Good practice for RNNs) ---
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
       
        optimizer.step()
        running_loss += loss.item() * bert_p.size(0)
        num_samples += bert_p.size(0)
       
    return running_loss / num_samples if num_samples > 0 else 0.0


def eval_model(model, loader, device):
    """Evaluates the model and computes metrics."""
    model.eval()
    preds_all = []
    labels_all = []
    with torch.no_grad():
        for bert_p, sent_p, mask, lengths, sent_agg, labels in loader:
            # Move all necessary tensors to device
            bert_p = bert_p.to(device)
            sent_p = sent_p.to(device)
            mask = mask.to(device)
            lengths = lengths.to(device)
            sent_agg = sent_agg.to(device)
            # labels remain on CPU for final metric concatenation

            preds = model(bert_p, sent_p, mask, lengths, sent_agg)
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.numpy())

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

# ====================================================================
# 4) MAIN EXECUTION BLOCK
# ====================================================================
if __name__ == "__main__":
   
    # 1. Load Data
    try:
        ordered_bert_seqs, ordered_sent_seqs, ordered_lengths, ordered_phq, bert_dim = load_features()
    except Exception as e:
        print(f"FATAL ERROR during data loading: {e}")
        sys.exit(1)

    full_dataset = SentenceLevelDataset(ordered_bert_seqs, ordered_sent_seqs, ordered_lengths, ordered_phq)
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # 2. Hyperparameter Search Setup
    keys, values = zip(*HYPERPARAM_GRID.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
   
    results = []
   
    # --- Global Tracking for BEST Model ---
    best_global_r = -1.0
    best_global_re = float("inf")
    best_global_state = None
    best_global_hparams = None
   
    start_time = time.time()
   
    # 3. K-Fold Cross-Validation and Hyperparameter Loop
    for fusion in FUSION_MODES:
        print("\n" + "="*60)
        print(f"🔥 Evaluating Fusion Mode: {fusion} 🔥")
        print("="*60)
        for combo in combos:
            hidden_size = combo["hidden_size"]
            lr = combo["lr"]
            dropout = combo["dropout"]
            batch_size = combo["batch_size"]
            num_layers = combo["num_layers"]

            fold_preds = []
            fold_labels = []

            print(f"\nGrid: H={hidden_size}, LR={lr}, D={dropout}, B={batch_size}, L={num_layers}")

            fold_idx = 0
            for train_idx_full, test_idx in kf.split(np.arange(len(full_dataset))):
                fold_idx += 1
                
                # --- Split Train into Train/Val (80/20) ---
                train_idx, val_idx = train_test_split(train_idx_full, test_size=0.2, random_state=RANDOM_STATE)

                # --- DataLoader Setup ---
                train_subset = torch.utils.data.Subset(full_dataset, train_idx)
                val_subset = torch.utils.data.Subset(full_dataset, val_idx)
                test_subset = torch.utils.data.Subset(full_dataset, test_idx)

                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
                test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

                # --- Model & Optimizer Setup ---
                model = ContextualLSTM(bert_dim=bert_dim, hidden_size=hidden_size,
                                       num_layers=num_layers, dropout=dropout, fusion=fusion).to(DEVICE)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
                criterion = nn.MSELoss()

                # --- Training Loop with Early Stopping ---
                best_val_loss = float("inf")
                patience_counter = 0
                current_best_fold_state = {k: v.cpu() for k, v in model.state_dict().items()} # Initial state

                for epoch in range(1, EPOCHS + 1):
                    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
                   
                    # Validation using VAL set for early stopping
                    r_val, re_val, _, _ = eval_model(model, val_loader, DEVICE)
                    val_metric = re_val # Using RE as the early stopping metric
                    
                    # Step scheduler
                    scheduler.step(val_metric)

                    if val_metric < best_val_loss:
                        best_val_loss = val_metric
                        patience_counter = 0
                        current_best_fold_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    else:
                        patience_counter += 1
                        if patience_counter >= PATIENCE:
                            break
                           
                # --- Post-Fold Evaluation ---
                if current_best_fold_state is not None:
                    model.load_state_dict(current_best_fold_state) # Load best checkpoint for final evaluation

                r_test, re_test, preds_test, labels_test = eval_model(model, test_loader, DEVICE)
                print(f" Fold {fold_idx} -- r: {r_test:.4f}, RE: {re_test:.4f}, n_test: {len(labels_test)}")
               
                fold_preds.append(preds_test)
                fold_labels.append(labels_test)

            # 4. Aggregate Fold Results for Combo
            all_preds = np.concatenate(fold_preds)
            all_labels = np.concatenate(fold_labels)
           
            if np.std(all_preds) == 0 or np.std(all_labels) == 0:
                agg_r = 0.0
            else:
                agg_r, _ = pearsonr(all_labels, all_preds)
            max_label_all = np.max(all_labels) if np.max(all_labels) != 0 else 1.0
            agg_re = np.mean(np.abs(all_preds - all_labels) / max_label_all)

            print(f"\n>>> GRID RESULT: Aggregate r: {agg_r:.4f}, Aggregate RE: {agg_re:.4f}")
           
            # 5. Global Best Check and Save
            if agg_r > best_global_r:
                best_global_r = agg_r
                best_global_re = agg_re
                # Store the model state that achieved this r (we'll save it later)
                # Since the current 'model' holds the best weights for the *last* fold,
                # we need to re-run the *best fold* if we wanted the single-best-fold model,
                # but for simplicity and standard practice, we only save the hparams and the aggregate metrics now.
                # NOTE: The current code structure prevents saving the *aggregate best model* easily,
                # so we will save the hyperparameters of the best aggregate model.
                best_global_hparams = {
                    "fusion": fusion,
                    "hidden_size": hidden_size,
                    "lr": lr,
                    "dropout": dropout,
                    "batch_size": batch_size,
                    "num_layers": num_layers,
                    "agg_r": float(agg_r),
                    "agg_re": float(agg_re)
                }

            results.append({
                "fusion": fusion,
                "hidden_size": hidden_size,
                "lr": lr,
                "dropout": dropout,
                "batch_size": batch_size,
                "num_layers": num_layers,
                "agg_r": float(agg_r),
                "agg_re": float(agg_re)
            })

    # ====================================================================
    # 5) FINAL REPORTING AND SAVING
    # ====================================================================
    elapsed = time.time() - start_time
   
    print("\n" + "=" * 60)
    print("HYPERPARAM SEARCH COMPLETE")
    print(f"Elapsed time: {elapsed/60:.2f} minutes")
    print("-" * 60)
   
    # Save best hyperparameters and final metrics
    best_hparam_path = os.path.join(MODEL_SAVE_DIR, "best_language_contextual_hyperparameters.json")
    if best_global_hparams:
        with open(best_hparam_path, "w") as f:
            json.dump(best_global_hparams, f, indent=4)
        print(f"🏆 Top result (Aggregate r={best_global_r:.4f}, RE={best_global_re:.4f}) found with:")
        print(json.dumps(best_global_hparams, indent=4))
        print(f"\nSaved best aggregate hyperparameters to: {best_hparam_path}")
       
        # --- How to Save the BEST Global Model (New Logic) ---
        # Since the best model is based on aggregate metrics across 5 folds,
        # we retrain a FINAL model on the entire dataset using the best hparams.
        # This is a standard way to get a single deployable model.
       
        print("\nRetraining BEST model on ALL data for final save...")
        final_model = ContextualLSTM(bert_dim=bert_dim,
                                     hidden_size=best_global_hparams["hidden_size"],
                                     num_layers=best_global_hparams["num_layers"],
                                     dropout=best_global_hparams["dropout"],
                                     fusion=best_global_hparams["fusion"]).to(DEVICE)
       
        final_optimizer = torch.optim.Adam(final_model.parameters(), lr=best_global_hparams["lr"], weight_decay=WEIGHT_DECAY)
        final_criterion = nn.MSELoss()
       
        # We train for a set number of epochs (e.g., the average epochs found, or the maximum EPOCНS)
        # Using the full dataset (all participants)
        final_loader = DataLoader(full_dataset, batch_size=best_global_hparams["batch_size"], shuffle=True, collate_fn=collate_fn)
       
        # Train for a full run (or max epochs for stability)
        for epoch in range(1, EPOCHS + 1):
             final_loss = train_epoch(final_model, final_loader, final_optimizer, final_criterion, DEVICE)
             if epoch % 10 == 0 or epoch == 1:
                 print(f" Final Train Epoch {epoch}/{EPOCHS} | Loss: {final_loss:.4f}")

        best_model_path = os.path.join(MODEL_SAVE_DIR, "contextual_lstm_best_overall.pt")
        torch.save(final_model.state_dict(), best_model_path)
        print(f"\nSaved FINAL BEST model (trained on all data) to: {best_model_path}")
   
    else:
        print("\nCould not find a stable model with positive correlation to save. Check data and model stability.")

    # Save all results to file for later inspection
    out_path = os.path.join(DATA_DIR, "contextual_lstm_sentence_results.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nAll combination results saved to: {out_path}")

    print("\nDone.")