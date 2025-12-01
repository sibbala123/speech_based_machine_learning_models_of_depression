#!/usr/bin/env python3
"""
Enhanced Test script for Advanced Acoustic LSTM model
"""

import os
import glob
import pickle
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
#                         GPU VALIDATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ============================================================
#                         LOCAL PATH CONFIG
# ============================================================
ACOUSTIC_DIR = r"C:\Users\jayan\Downloads\E-DAIC_Acoustics-20251116T231610Z-1-001\E-DAIC_Acoustics"
LABELS_PATH  = r"C:\Users\jayan\Downloads\DepressionLabels.xlsx"

MODEL_SAVE_DIR = r"C:\Users\jayan\ML_Projects\speech_based_machine_learning_models_of_depression\models"

# File paths for loading
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "lstm_acoustic_best_overall.pt")
BEST_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, "best_acoustic_scaler.pkl")
BEST_HYPERPARAM_PATH = os.path.join(MODEL_SAVE_DIR, "best_Acoustic_Hyperparameters.json")
FINAL_TEST_RESULTS_PATH = os.path.join(MODEL_SAVE_DIR, "acoustic_final_test_results.json")

# Fixed experiment settings
RANDOM_STATE = 42
FINAL_TEST_SIZE = 14

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
#                 LOAD ACOUSTIC FEATURE FILES & LABELS
# ============================================================
def extract_pid_from_filename(path):
    return os.path.basename(path).split("_")[0]

print("Starting Enhanced Acoustic Model Final Test Evaluation...")
print("\nScanning acoustic files in:", ACOUSTIC_DIR)
acoustic_files = sorted(glob.glob(os.path.join(ACOUSTIC_DIR, "*.csv")))
if not acoustic_files:
    raise FileNotFoundError(f"No CSV files found in {ACOUSTIC_DIR}")

print("Loading labels:", LABELS_PATH)
labels_df = pd.read_excel(LABELS_PATH)
possible_cols = [c for c in labels_df.columns if "participant" in c.lower()]
if not possible_cols:
    raise RuntimeError("No participant column found in labels file.")
labels_df.rename(columns={possible_cols[0]:"participant"}, inplace=True)
labels_df["participant"] = labels_df["participant"].astype(str)

phq_col = next((c for c in labels_df.columns if "phq" in c.lower()), None)
if phq_col is None:
    num_cols = labels_df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        raise RuntimeError("Could not find PHQ column or any numeric columns in labels.")
    phq_col = num_cols[0]

# --- Build Sequences ---
file_map = {extract_pid_from_filename(f): f for f in acoustic_files}
participant_ids = []
sequences_dict = {}
sequence_lengths_dict = {}
phq_scores_dict = {}
COLUMNS_TO_DROP = ['frameTime', 'duration', 'Unnamed: 0']

for pid, path in file_map.items():
    if pid not in labels_df["participant"].values:
        continue

    df = pd.read_csv(path)
    df_num = df.select_dtypes(include="number").drop(columns=[col for col in COLUMNS_TO_DROP if col in df.columns], errors='ignore')

    if df_num.empty: continue
    if df_num.isnull().values.any():
        df_num = df_num.fillna(0) 

    arr = df_num.values.astype(np.float32)

    sequences_dict[pid] = arr
    sequence_lengths_dict[pid] = arr.shape[0]
    participant_ids.append(pid)
    phq_scores_dict[pid] = float(labels_df.loc[labels_df["participant"] == pid, phq_col].values[0])

ordered_pids_all = participant_ids
num_features = arr.shape[1] if sequences_dict else 0
if not ordered_pids_all:
    sys.exit("Error: No valid participant data loaded.")

print(f"Total participants loaded: {len(ordered_pids_all)}, features per frame: {num_features}")


# ============================================================
#        DATASET + COLLATE FN
# ============================================================
class AcousticSeqDataset(Dataset):
    def __init__(self, seqs, lengths, labels):
        self.seqs = seqs
        self.lengths = lengths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.tensor(self.seqs[idx], dtype=torch.float32)
        L = self.lengths[idx]
        return seq, L, float(self.labels[idx])


def collate_acoustic(batch):
    seqs, lengths, labels = zip(*batch)
    lengths = torch.tensor(lengths, dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True)
    mask = (torch.arange(padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)).float()
    labels = torch.tensor(labels, dtype=torch.float32)
    return padded, lengths, mask, labels


# ============================================================
#          ADVANCED MODEL WITH ATTENTION
# ============================================================
class AttentionLayer(nn.Module):
    """Attention mechanism over time steps"""
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_out, mask):
        attn_weights = self.attention(lstm_out).squeeze(-1)
        attn_weights = attn_weights.masked_fill(mask.to(lstm_out.device) == 0, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=1).unsqueeze(-1)
        context = torch.sum(lstm_out * attn_weights, dim=1)
        return context, attn_weights


class EnhancedLSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = num_layers
        
        self.input_dropout = nn.Dropout(dropout * 0.5)
        
        self.lstm = nn.LSTM(
            input_dim, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.layer_norm = nn.LayerNorm(hidden_size * self.num_directions)
        self.attention = AttentionLayer(hidden_size * self.num_directions)
        
        self.fc1 = nn.Linear(hidden_size * self.num_directions, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc_out = nn.Linear(64, 1)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x, lengths, mask):
        x = self.input_dropout(x)
        
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        
        out = self.layer_norm(out)
        context, _ = self.attention(out, mask)
        
        x = self.fc1(context)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        return self.fc_out(x).squeeze(1)


# ============================================================
#                      EVAL UTILITY
# ============================================================
def evaluate(model, loader):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for padded, lengths, mask, labels in loader:
            padded, lengths = padded.to(DEVICE), lengths.to(DEVICE)
            pred = model(padded, lengths, mask).cpu().numpy()
            preds.append(pred)
            trues.append(labels.numpy())

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
        return np.nan, np.nan, preds, trues

    if np.std(preds) == 0 or np.std(trues) == 0:
        r = 0.0
    else:
        r, _ = pearsonr(trues, preds)

    max_label = np.max(trues) if np.max(trues) != 0 else 1.0
    re = np.mean(np.abs(preds - trues) / max_label)

    return r, re, preds, trues


# ============================================================
#                       MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    
    # Split data to get the same test set as training
    all_indices = np.arange(len(ordered_pids_all))
    train_cv_idx, final_test_idx, _, _ = train_test_split(
        all_indices, all_indices, 
        test_size=FINAL_TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    pids_test = [ordered_pids_all[i] for i in final_test_idx]
    sequences_test = [sequences_dict[p] for p in pids_test]
    lengths_test = [sequence_lengths_dict[p] for p in pids_test]
    phq_test = np.array([phq_scores_dict[p] for p in pids_test], dtype=np.float32)
    
    print(f"Identified {len(pids_test)} participants for final testing (N={FINAL_TEST_SIZE}).")
    
    # Load best hyperparameters
    try:
        with open(BEST_HYPERPARAM_PATH, "r") as f:
            best_combo_info = json.load(f)
        best_hparams = best_combo_info['hyperparams']
        print(f"Loaded Best Hparams: Hidden={best_hparams['hidden_size']}, Layers={best_hparams['num_layers']}, Dropout={best_hparams['dropout']:.2f}")
    except FileNotFoundError:
        print(f"Error: Hyperparameter file not found at {BEST_HYPERPARAM_PATH}. Please run the training script first.")
        sys.exit(1)
    
    # Load scaler
    try:
        with open(BEST_SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print(f"Loaded scaler from: {BEST_SCALER_PATH}")
    except FileNotFoundError:
        print(f"Error: Scaler file not found at {BEST_SCALER_PATH}. Please run the training script first.")
        sys.exit(1)
    
    # Scale test data
    seq_test_scaled = [scaler.transform(s) for s in sequences_test]
    
    # Prepare DataLoader
    ds_test = AcousticSeqDataset(seq_test_scaled, lengths_test, phq_test)
    test_batch_size = best_hparams['batch_size']
    test_loader = DataLoader(ds_test, batch_size=test_batch_size, shuffle=False, collate_fn=collate_acoustic)
    
    # Initialize Model and Load BEST Weights
    H_best = best_hparams['hidden_size']
    D_best = best_hparams['dropout']
    L_best = best_hparams['num_layers']
    
    final_model = EnhancedLSTMRegressor(num_features, H_best, L_best, D_best, bidirectional=True).to(DEVICE)
    
    try:
        final_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model weights from: {BEST_MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model weights not found at {BEST_MODEL_PATH}. Please run the training script first.")
        sys.exit(1)
    
    # Run Evaluation
    r_final, re_final, preds_final, trues_final = evaluate(final_model, test_loader)
    
    print("\n" + "="*60)
    print("*** Enhanced Acoustic Final Holdout Test Performance (N=14) ***")
    print(f"  Pearson's r: **{r_final:.4f}**")
    print(f"  Relative Error (RE): **{re_final:.4f}**")
    print("="*60)
    
    # Save Final Test Results
    final_test_summary = {
        "model": "Enhanced Acoustic LSTM with Attention",
        "N_test": len(pids_test),
        "r_test": float(r_final) if np.isfinite(r_final) else None,
        "re_test": float(re_final) if np.isfinite(re_final) else None,
        "test_pids": pids_test,
        "true_scores": trues_final.tolist(),
        "predicted_scores": preds_final.tolist(),
        "best_cv_hyperparams": best_hparams
    }
    with open(FINAL_TEST_RESULTS_PATH, "w") as f:
        json.dump(final_test_summary, f, indent=4)
    print(f"\nFinal test results saved to: {FINAL_TEST_RESULTS_PATH}")
    
    print("Done.")
