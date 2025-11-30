#!/usr/bin/env python3
"""
acoustic_unimodal_complete_pipeline.py — V4: Combines K-Fold CV for 
hyperparameter tuning AND a final evaluation on a holdout test set (N=14).
"""

import os
import glob
import pickle
import sys
import json
import itertools
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from sklearn.model_selection import KFold, train_test_split # Added train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================
#                         GPU VALIDATION
# ============================================================
print("\n================ GPU CHECK ================")
if torch.cuda.is_available():
    print("CUDA is available.")
    print("GPU device:", torch.cuda.get_device_name(0))
    DEVICE = torch.device("cuda")
else:
    print("CUDA NOT available. Using CPU.")
    DEVICE = torch.device("cpu")
print("===========================================\n")


# ============================================================
#                         LOCAL PATH CONFIG
# ============================================================
ACOUSTIC_DIR = r"C:\Users\jayan\Downloads\E-DAIC_Acoustics-20251116T231610Z-1-001\E-DAIC_Acoustics"
LABELS_PATH  = r"C:\Users\jayan\Downloads\DepressionLabels.xlsx"
OUTPUT_FEATURES_PKL = r"C:\Users\jayan\ML_Projects\speech_based_machine_learning_models_of_depression\features\acoustic_features.pkl"

MODEL_SAVE_DIR = r"C:\Users\jayan\ML_Projects\speech_based_machine_learning_models_of_depression\models"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# File names for saving results:
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "lstm_acoustic_best_overall.pt")
BEST_SCALER_PATH = os.path.join(MODEL_SAVE_DIR, "best_acoustic_scaler.pkl") # Renamed to avoid collision
BEST_HYPERPARAM_PATH = os.path.join(MODEL_SAVE_DIR, "best_Acoustic_Hyperparameters.json")
FINAL_TEST_RESULTS_PATH = os.path.join(MODEL_SAVE_DIR, "acoustic_final_test_results.json")


# ============================================================
#                         TRAINING CONFIG
# ============================================================
NUM_FOLDS = 5
RANDOM_STATE = 42
FINAL_TEST_SIZE = 14 # The N=14 holdout set size

# default hyperparams (will be overridden by grid)
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.3
BATCH_SIZE = 16
LR = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 60
PATIENCE = 8  # early stopping
GRAD_CLIP_NORM = 1.0 # Gradient Clipping

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
#                  HYPERPARAMETER GRID (SMALL GRID)
# ============================================================
HIDDEN_SIZE_GRID = [64, 128]
DROPOUT_GRID     = [0.2, 0.3]
LR_GRID          = [1e-3, 5e-4]
BATCH_SIZE_GRID  = [8, 16]

grid_combinations = list(itertools.product(HIDDEN_SIZE_GRID, DROPOUT_GRID, LR_GRID, BATCH_SIZE_GRID))


# ============================================================
#                 LOAD ACOUSTIC FEATURE FILES & LABELS (Integrated)
# ============================================================
def extract_pid_from_filename(path):
    return os.path.basename(path).split("_")[0]

print("Scanning acoustic files in:", ACOUSTIC_DIR)
acoustic_files = sorted(glob.glob(os.path.join(ACOUSTIC_DIR, "*.csv")))
if not acoustic_files:
    raise FileNotFoundError(f"No CSV files found in {ACOUSTIC_DIR}")

print("\nLoading labels:", LABELS_PATH)
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

print("Using PHQ column:", phq_col)

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
#        DATASET + COLLATE FN (Unchanged)
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
#                         MODEL DEF (Unchanged)
# ============================================================
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc_out = nn.Linear(64, 1)

    def forward(self, x, lengths, mask):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = out * mask.unsqueeze(-1).to(out.device)
        last_idx = (lengths - 1).view(-1, 1, 1)
        last = out.gather(1, last_idx.expand(out.size(0), 1, out.size(2))).squeeze(1)
        x = torch.relu(self.fc1(last))
        return self.fc_out(x).squeeze(1)


# ============================================================
#                      TRAIN / EVAL UTILITIES (Unchanged)
# ============================================================
def train_epoch(model, loader, optimizer, criterion, grad_clip_norm):
    model.train()
    total = 0
    loss_sum = 0

    for padded, lengths, mask, labels in loader:
        padded, lengths, labels = (
            padded.to(DEVICE),
            lengths.to(DEVICE),
            labels.to(DEVICE),
        )

        optimizer.zero_grad()
        preds = model(padded, lengths, mask)
        loss = criterion(preds, labels)

        if torch.isnan(loss):
            print("CRITICAL: NaN loss detected. Skipping batch.")
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        loss_sum += loss.item() * len(labels)
        total += len(labels)

    return loss_sum / total if total > 0 else np.nan


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
#                       MAIN EXECUTION START
# ============================================================
if __name__ == "__main__":
    
    # 1. INITIAL TRAIN/TEST SPLIT (Splits all data into CV set and a final Test set)
    print("\n================ DATA SPLIT ================")
    # Split PIDs into the large train/CV set and the small final holdout test set
    pids_train_cv, pids_test, _, _ = train_test_split(
        ordered_pids_all, ordered_pids_all, 
        test_size=FINAL_TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    # Prepare data arrays for the train/CV split (only the data that will be used for tuning)
    sequences_train_cv = [sequences_dict[p] for p in pids_train_cv]
    lengths_train_cv = [sequence_lengths_dict[p] for p in pids_train_cv]
    phq_train_cv = np.array([phq_scores_dict[p] for p in pids_train_cv], dtype=np.float32)
    
    print(f"Total Data: N={len(ordered_pids_all)}")
    print(f"Train/CV Data: N={len(pids_train_cv)}")
    print(f"Final Test Data: N={len(pids_test)}")
    print("============================================")

    # ============================================================
    # 2. HYPERPARAMETER SEARCH (CV on Train/CV Data)
    # ============================================================
    print("\n================ HYPERPARAMETER SEARCH (CV) ================")
    
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_combo_re = float("inf")
    best_combo_info = None
    all_results = []
    best_model_state_for_save = None
    best_scaler_for_save = None

    start_time = time.time()
    total_combos = len(grid_combinations)
    
    # --- Start CV Loop (Original Logic) ---
    for combo_idx, (HIDDEN_SIZE_c, DROPOUT_c, LR_c, BATCH_SIZE_c) in enumerate(grid_combinations, start=1):
        combo_start = time.time()
        print("\n" + "="*60)
        print(f"Combo {combo_idx}/{total_combos} — HIDDEN={HIDDEN_SIZE_c}, DROPOUT={DROPOUT_c}, LR={LR_c}, BATCH={BATCH_SIZE_c}")
        print("="*60)

        fold_re_list = []
        combo_best_fold_re = float("inf")
        combo_best_fold_state = None
        combo_best_scaler = None
        combo_best_fold_details = None
        
        # Use the TRAIN/CV data for KFold
        for fold, (train_idx, test_idx) in enumerate(kf.split(sequences_train_cv), start=1):
            print(f"\n--- Combo {combo_idx} | Fold {fold} ---")
            
            # Extract data for this fold
            seq_train = [sequences_train_cv[i] for i in train_idx]
            seq_test = [sequences_train_cv[i] for i in test_idx]
            len_train = [lengths_train_cv[i] for i in train_idx]
            len_test = [lengths_train_cv[i] for i in test_idx]
            y_train = phq_train_cv[train_idx]
            y_test = phq_train_cv[test_idx]

            # SCALER FIT on train only
            flattened = np.vstack(seq_train)
            scaler = StandardScaler().fit(flattened)

            seq_train_scaled = [scaler.transform(s) for s in seq_train]
            seq_test_scaled = [scaler.transform(s) for s in seq_test]

            train_ds = AcousticSeqDataset(seq_train_scaled, len_train, y_train)
            test_ds = AcousticSeqDataset(seq_test_scaled, len_test, y_test)

            train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_c, shuffle=True, collate_fn=collate_acoustic)
            test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE_c, shuffle=False, collate_fn=collate_acoustic)

            model = LSTMRegressor(num_features, HIDDEN_SIZE_c, NUM_LAYERS, DROPOUT_c).to(DEVICE)
            optim = torch.optim.Adam(model.parameters(), lr=LR_c, weight_decay=WEIGHT_DECAY)
            criterion = nn.MSELoss()

            best_val_re = float("inf")
            patience = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()} # Initial state save

            for epoch in range(1, EPOCHS + 1):
                loss = train_epoch(model, train_loader, optim, criterion, GRAD_CLIP_NORM)
                r_val, re_val, _, _ = evaluate(model, test_loader)

                loss_str = f"{loss:.4f}" if not np.isnan(loss) else "nan"
                r_str = f"{r_val:.4f}" if not np.isnan(r_val) else "nan"
                re_str = f"{re_val:.4f}" if not np.isnan(re_val) else "nan"
                print(f"Combo {combo_idx} | Fold {fold} | Epoch {epoch} | Loss {loss_str} | r {r_str} | RE {re_str}")

                if np.isnan(loss) or np.isnan(re_val):
                    print(f"Warning: NaN detected. Stopping fold {fold} early.")
                    break

                if re_val < best_val_re:
                    best_val_re = re_val
                    patience = 0
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= PATIENCE:
                        print(f"Early stopping at epoch {epoch} for fold {fold}.")
                        break

            # Evaluate best model state from this fold
            model.load_state_dict(best_state)
            r_test, re_test, _, _ = evaluate(model, test_loader)

            print(f"Combo {combo_idx} | Fold {fold} final -> r = {r_test:.4f} | RE = {re_test:.4f}")

            fold_re_list.append(float(re_test) if np.isfinite(re_test) else float("inf"))

            # track best fold state for this combo
            if np.isfinite(re_test) and re_test < combo_best_fold_re:
                combo_best_fold_re = re_test
                combo_best_fold_state = {k: v.cpu() for k, v in model.state_dict().items()}
                combo_best_scaler = scaler
                combo_best_fold_details = {
                    "fold": fold,
                    "r_test": float(r_test),
                    "re_test": float(re_test),
                }

        # After all folds for this combo
        mean_re = float(np.mean([x for x in fold_re_list if np.isfinite(x)]))
        std_re = float(np.std([x for x in fold_re_list if np.isfinite(x)]))

        combo_result = {
            "hidden_size": int(HIDDEN_SIZE_c),
            "dropout": float(DROPOUT_c),
            "lr": float(LR_c),
            "batch_size": int(BATCH_SIZE_c),
            "fold_re_list": [float(x) if np.isfinite(x) else None for x in fold_re_list],
            "mean_re": mean_re,
            "std_re": std_re,
            "best_fold_re": float(combo_best_fold_re) if np.isfinite(combo_best_fold_re) else None,
            "best_fold_details": combo_best_fold_details,
        }

        all_results.append(combo_result)

        print(f"\nCombo {combo_idx} summary: mean RE = {mean_re:.4f} | std RE = {std_re:.4f}")

        # Update global best combo if improved
        if mean_re < best_combo_re:
            best_combo_re = mean_re
            best_combo_info = combo_result
            best_model_state_for_save = combo_best_fold_state
            best_scaler_for_save = combo_best_scaler
            best_combo_info["hyperparams"] = {
                "hidden_size": int(HIDDEN_SIZE_c),
                "num_layers": int(NUM_LAYERS),
                "dropout": float(DROPOUT_c),
                "batch_size": int(BATCH_SIZE_c),
                "lr": float(LR_c),
                "weight_decay": float(WEIGHT_DECAY),
                "epochs": int(EPOCHS),
                "patience": int(PATIENCE),
                "grad_clip_norm": float(GRAD_CLIP_NORM),
                "num_folds": int(NUM_FOLDS),
                "random_state": int(RANDOM_STATE),
            }

        combo_end = time.time()
        print(f"Combo {combo_idx} time: {combo_end - combo_start:.1f}s")

    total_time = time.time() - start_time
    print(f"\nHyperparameter search complete in {total_time/60:.2f} minutes.")

    # ============================================================
    # 3. SAVE BEST RESULTS FROM CV
    # ============================================================
    if best_combo_info is not None and best_model_state_for_save is not None:
        # Save model state dict
        torch.save(best_model_state_for_save, BEST_MODEL_PATH)
        print(f"\nSaved BEST CV model: {BEST_MODEL_PATH}")

        # Save scaler
        pickle.dump(best_scaler_for_save, open(BEST_SCALER_PATH, "wb"))
        print(f"Saved BEST CV scaler: {BEST_SCALER_PATH}")

        # Save best hyperparams JSON
        with open(BEST_HYPERPARAM_PATH, "w") as f:
            json.dump(best_combo_info, f, indent=4)
        print(f"Saved BEST CV hyperparameters to: {BEST_HYPERPARAM_PATH}")
        
        # Save all results JSON
        all_results_path = os.path.join(MODEL_SAVE_DIR, "all_Acoustic_Hyperparameter_Results.json")
        with open(all_results_path, "w") as f:
            json.dump(all_results, f, indent=4)
        print(f"Saved ALL hyperparameter results to: {all_results_path}")

    else:
        sys.exit("Could not find a stable model to save. Check data and model stability.")


    # ============================================================
    # 4. FINAL TEST EVALUATION (New Step)
    # ============================================================
    print("\n================ FINAL TEST EVALUATION ================")
    
    # 4a. Load the specific Test Set data
    sequences_test = [sequences_dict[p] for p in pids_test]
    lengths_test = [sequence_lengths_dict[p] for p in pids_test]
    phq_test = np.array([phq_scores_dict[p] for p in pids_test], dtype=np.float32)

    # 4b. Use the BEST SCALER found during CV
    scaler = best_scaler_for_save
    seq_test_scaled = [scaler.transform(s) for s in sequences_test]

    # 4c. Prepare DataLoader
    ds_test = AcousticSeqDataset(seq_test_scaled, lengths_test, phq_test)
    # Use the best batch size from the best combo, or default to 16 if not found
    test_batch_size = best_combo_info['hyperparams']['batch_size'] if best_combo_info else 16
    test_loader = DataLoader(ds_test, batch_size=test_batch_size, shuffle=False, collate_fn=collate_acoustic)

    # 4d. Initialize Model and Load BEST Weights
    H_best = best_combo_info['hyperparams']['hidden_size']
    D_best = best_combo_info['hyperparams']['dropout']
    L_best = best_combo_info['hyperparams']['num_layers']
    
    final_model = LSTMRegressor(num_features, H_best, L_best, D_best).to(DEVICE)
    final_model.load_state_dict(best_model_state_for_save)
    print(f"Loaded best model (H={H_best}, D={D_best}) for final test.")

    # 4e. Run Evaluation
    r_final, re_final, preds_final, trues_final = evaluate(final_model, test_loader)

    print("\n*** Acoustic Final Holdout Test Performance (N=14) ***")
    print(f"  Pearson's r: **{r_final:.4f}**")
    print(f"  Relative Error (RE): **{re_final:.4f}**")

    # 4f. Save Final Test Results
    final_test_summary = {
        "model": "Acoustic LSTM",
        "N_test": len(pids_test),
        "r_test": float(r_final) if np.isfinite(r_final) else None,
        "re_test": float(re_final) if np.isfinite(re_final) else None,
        "test_pids": pids_test,
        "true_scores": trues_final.tolist(),
        "predicted_scores": preds_final.tolist(),
        "best_cv_hyperparams": best_combo_info['hyperparams']
    }
    with open(FINAL_TEST_RESULTS_PATH, "w") as f:
        json.dump(final_test_summary, f, indent=4)
    print(f"Saved FINAL test results to: {FINAL_TEST_RESULTS_PATH}")
    
    print("\nTraining, Tuning, and Final Evaluation Complete.")