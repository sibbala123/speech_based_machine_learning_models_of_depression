#!/usr/bin/env python3
"""
train_multimodal_fusion.py
Combines pre-trained language and acoustic LSTM models using neural fusion.
Simplified version - loads pre-trained models and adds fusion layers on top.

*** MODIFIED: Added an initial train/test split and saving of CV predictions. ***
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
import itertools
import time
import json

# ====================================================================
# CONFIGURATION
# ====================================================================
DATA_DIR = r"C:\Users\jayan\ML_Projects\speech_based_machine_learning_models_of_depression\features"
MODEL_DIR = r"C:\Users\jayan\ML_Projects\speech_based_machine_learning_models_of_depression\models"

# Input files
LANGUAGE_PKL = os.path.join(DATA_DIR, "language_features_sentence_level.pkl")
ACOUSTIC_PKL = os.path.join(DATA_DIR, "acoustic_features.pkl")
LANGUAGE_MODEL = os.path.join(MODEL_DIR, "contextual_lstm_best_overall.pt")
ACOUSTIC_MODEL = os.path.join(MODEL_DIR, "lstm_acoustic_best_overall.pt")
LANGUAGE_HPARAMS = os.path.join(MODEL_DIR, "best_language_contextual_hyperparameters.json")
ACOUSTIC_SCALER = os.path.join(MODEL_DIR, "best_scaler.pkl")

# Output
FUSION_MODEL_DIR = os.path.join(MODEL_DIR, "multimodal")
os.makedirs(FUSION_MODEL_DIR, exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

# Training settings
NUM_FOLDS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.1 # New: 10% held-out test set
EPOCHS = 50
PATIENCE = 10
WEIGHT_DECAY = 1e-5

# Simplified hyperparameter grid
FUSION_GRID = {
    "fusion_hidden": [128, 256],
    "dropout": [0.3, 0.5],
    "lr": [1e-4, 5e-4],
    "batch_size": [16, 32]
}

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ====================================================================
# MODEL ARCHITECTURES (Same as training scripts)
# ====================================================================

class ContextualLSTM(nn.Module):
    """Language LSTM from training script."""
    def __init__(self, bert_dim=384, hidden_size=128, num_layers=2, dropout=0.3, fusion="concat"):
        super().__init__()
        self.fusion = fusion
        self.hidden_size = hidden_size
        
        input_size = bert_dim + 1 if fusion == "concat" else bert_dim
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        if fusion in ("gated", "attention"):
            self.sent_agg_encoder = nn.Sequential(
                nn.Linear(2, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, hidden_size))
        if fusion == "gated":
            self.gate_layer = nn.Linear(hidden_size, hidden_size)
        if fusion == "attention":
            self.att_text = nn.Linear(hidden_size, hidden_size)
            self.att_sent = nn.Linear(hidden_size, hidden_size)
            self.att_comb = nn.Linear(hidden_size, 1)
        
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def get_embedding(self, bert_seq, sent_seq, mask, lengths, sent_agg):
        """Extract embedding before final classification layers."""
        B, T, D = bert_seq.shape
        x = torch.cat([bert_seq, sent_seq], dim=2) if self.fusion == "concat" else bert_seq
        
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h_n, _) = self.lstm(packed)
        
        if self.fusion == "concat":
            return h_n[-1]  # Last layer hidden state
        else:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)
            # Use the hidden state corresponding to the last element of the sequence
            idx = (lengths - 1).long().unsqueeze(1).unsqueeze(2).expand(B, 1, self.hidden_size)
            h_last = lstm_out.gather(1, idx).squeeze(1)
            
            if self.fusion == "gated":
                sent_vec = self.sent_agg_encoder(sent_agg)
                gate = torch.sigmoid(self.gate_layer(sent_vec))
                return h_last * gate
            else:  # attention
                sent_vec = self.sent_agg_encoder(sent_agg).unsqueeze(1)
                att_in = torch.tanh(self.att_text(lstm_out) + self.att_sent(sent_vec))
                att_scores = self.att_comb(att_in).squeeze(-1).masked_fill(mask == 0, float("-inf"))
                att_weights = torch.softmax(att_scores, dim=1).unsqueeze(-1)
                return torch.sum(lstm_out * att_weights, dim=1)


class AcousticLSTM(nn.Module):
    """Acoustic LSTM from training script."""
    def __init__(self, input_dim, hidden_size=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc_out = nn.Linear(64, 1)

    def get_embedding(self, x, lengths, mask):
        """Extract embedding before final classification layers."""
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)
        
        out = out * mask.unsqueeze(-1)
        # Get the final hidden state before padding starts
        last_idx = (lengths - 1).view(-1, 1, 1).expand(out.size(0), 1, out.size(2))
        return out.gather(1, last_idx).squeeze(1)


# ====================================================================
# FUSION MODEL
# ====================================================================

class MultimodalFusion(nn.Module):
    """
    Combines language and acoustic embeddings.
    Both branches are fine-tuned during training.
    """
    def __init__(self, lang_model, acoustic_model, fusion_hidden=128, dropout=0.3):
        super().__init__()
        self.lang_model = lang_model
        self.acoustic_model = acoustic_model
        
        # Get embedding dimensions
        lang_dim = lang_model.hidden_size
        acoustic_dim = acoustic_model.lstm.hidden_size
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(lang_dim + acoustic_dim, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
    
    def forward(self, lang_inputs, acoustic_inputs):
        """
        lang_inputs: (bert_seq, sent_seq, mask, lengths, sent_agg)
        acoustic_inputs: (acoustic_seq, lengths, mask)
        """
        bert_seq, sent_seq, lang_mask, lang_lengths, sent_agg = lang_inputs
        acoustic_seq, acoustic_lengths, acoustic_mask = acoustic_inputs
        
        # Extract embeddings from both models
        lang_embed = self.lang_model.get_embedding(bert_seq, sent_seq, lang_mask, 
                                                     lang_lengths, sent_agg)
        acoustic_embed = self.acoustic_model.get_embedding(acoustic_seq, acoustic_lengths, 
                                                             acoustic_mask)
        
        # Concatenate and pass through fusion layers
        combined = torch.cat([lang_embed, acoustic_embed], dim=1)
        return self.fusion(combined).squeeze(1)


# ====================================================================
# DATA LOADING
# ====================================================================

def load_and_align_data():
    """Load both datasets and find common participants."""
    print("\nLoading language features...")
    with open(LANGUAGE_PKL, "rb") as f:
        lang_data = pickle.load(f)
    
    print("Loading acoustic features...")
    with open(ACOUSTIC_PKL, "rb") as f:
        acoustic_data = pickle.load(f)
    
    # --- Alignment Logic ---
    # Find participant IDs from the keys of the feature dictionaries
    if isinstance(lang_data["bert_sequences"], dict):
        lang_pids_raw = list(lang_data["bert_sequences"].keys())
        # Assuming lang_data["phq_scores"] is indexed by lang_data["participant_ids"]
        # We need a map from PID to score for lookup
        phq_pid_to_score = {str(pid).strip(): score for pid, score in zip(lang_data["participant_ids"], lang_data["phq_scores"])}
    else:
        # Fallback to the original logic if it's a list
        lang_pids_raw = lang_data["participant_ids"]
        phq_pid_to_score = {str(pid).strip(): score for pid, score in zip(lang_data["participant_ids"], lang_data["phq_scores"])}
        
    acoustic_pids_raw = acoustic_data["participant_ids"]
    
    # Normalize IDs to strings and strip whitespace
    lang_pids = set(str(pid).strip() for pid in lang_pids_raw)
    acoustic_pids = set(str(pid).strip() for pid in acoustic_pids_raw)
    common_pids = sorted(lang_pids & acoustic_pids)
    
    print(f"\nData Alignment: Common participants: {len(common_pids)}")
    if len(common_pids) == 0:
        raise ValueError("No common participants! Check participant ID formats.")
    
    # === NEWLY ADDED/REQUIRED SECTION FOR ACOUSTIC MAPS ===
    # Create PID to feature map for acoustic data for easier lookup
    acoustic_pid_to_seq = {str(pid).strip(): seq for pid, seq in zip(acoustic_data["participant_ids"], acoustic_data["sequences"])}
    acoustic_pid_to_len = {str(pid).strip(): length for pid, length in zip(acoustic_data["participant_ids"], acoustic_data["sequence_lengths"])}
    # =======================================================

    # Align and combine data
    # Convert common_pids (which are strings) to integers for language data lookup
    # We assume the acoustic data keys are strings, so we keep the strings for acoustic lookup.
    common_pids_int = [int(pid) for pid in common_pids]

    # Use the integer PIDs for lookup in lang_data["bert_sequences"] (which has int keys)
    lang_bert = [lang_data["bert_sequences"][pid] for pid in common_pids_int]
    lang_sent = [lang_data["sentiment_sequences"][pid] for pid in common_pids_int]
    lang_lengths = [len(lang_bert[i]) for i in range(len(common_pids))]

    # Use the string PIDs for acoustic data lookup
    acoustic_seqs = [acoustic_pid_to_seq[pid] for pid in common_pids]
    acoustic_lengths = [acoustic_pid_to_len[pid] for pid in common_pids]

    phq_scores = np.array([phq_pid_to_score[pid] for pid in common_pids], dtype=np.float32)

    return common_pids, lang_bert, lang_sent, lang_lengths, acoustic_seqs, acoustic_lengths, phq_scores


class MultimodalDataset(Dataset):
    """Combined dataset."""
    def __init__(self, pids, lang_bert, lang_sent, lang_lengths, acoustic_seqs, acoustic_lengths, labels):
        self.pids = pids # New: Store PIDs
        self.lang_bert = lang_bert
        self.lang_sent = lang_sent
        self.lang_lengths = lang_lengths
        self.acoustic_seqs = acoustic_seqs
        self.acoustic_lengths = acoustic_lengths
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.pids[idx], # New: Return PID
            np.stack(self.lang_bert[idx]).astype(np.float32),
            np.array(self.lang_sent[idx], dtype=np.float32),
            int(self.lang_lengths[idx]),
            torch.tensor(self.acoustic_seqs[idx], dtype=torch.float32),
            int(self.acoustic_lengths[idx]),
            float(self.labels[idx])
        )


def collate_fn(batch):
    """Collate function for batching."""
    pids, lang_bert, lang_sent, lang_len, acoustic_seq, acoustic_len, labels = zip(*batch)
    
    # Language data
    lang_len = np.array(lang_len, dtype=np.int64)
    max_lang = int(lang_len.max())
    B = len(batch)
    bert_dim = lang_bert[0].shape[1]
    
    bert_pad = np.zeros((B, max_lang, bert_dim), dtype=np.float32)
    sent_pad = np.zeros((B, max_lang), dtype=np.float32)
    lang_mask = np.zeros((B, max_lang), dtype=np.float32)
    
    for i, (b, s, L) in enumerate(zip(lang_bert, lang_sent, lang_len)):
        bert_pad[i, :L, :] = b
        sent_pad[i, :L] = s
        lang_mask[i, :L] = 1.0
    
    bert_pad = torch.from_numpy(bert_pad)
    sent_pad = torch.from_numpy(sent_pad).unsqueeze(-1)
    lang_mask = torch.from_numpy(lang_mask)
    lang_len = torch.from_numpy(lang_len)
    
    # Sentiment aggregation (mean and std)
    mask_sum = lang_mask.sum(dim=1)
    sent_mean = (sent_pad.squeeze(-1) * lang_mask).sum(dim=1) / (mask_sum + 1e-8)
    sent_std = torch.sqrt(((sent_pad.squeeze(-1) - sent_mean.unsqueeze(1))**2 * lang_mask).sum(dim=1) / (mask_sum + 1e-8))
    sent_agg = torch.stack([sent_mean, sent_std], dim=1)
    
    # Acoustic data
    acoustic_len = torch.tensor(acoustic_len, dtype=torch.long)
    acoustic_pad = nn.utils.rnn.pad_sequence(acoustic_seq, batch_first=True)
    acoustic_mask = (torch.arange(acoustic_pad.size(1)).unsqueeze(0) < acoustic_len.unsqueeze(1)).float()
    
    labels = torch.tensor(labels, dtype=torch.float32)
    
    return pids, (bert_pad, sent_pad, lang_mask, lang_len, sent_agg), (acoustic_pad, acoustic_len, acoustic_mask), labels


# ====================================================================
# TRAINING
# ====================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    num_samples = 0
    
    for _, lang_inputs, acoustic_inputs, labels in loader: # Ignore PIDs
        lang_inputs = tuple(x.to(device) for x in lang_inputs)
        acoustic_inputs = tuple(x.to(device) for x in acoustic_inputs)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        preds = model(lang_inputs, acoustic_inputs)
        loss = criterion(preds, labels)
        
        if torch.isnan(loss):
            continue
        
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
        num_samples += len(labels)
    
    return total_loss / num_samples if num_samples > 0 else 0.0


def evaluate_with_pids(model, loader, device):
    """Evaluate model and collect predictions along with PIDs."""
    model.eval()
    preds_all, labels_all, pids_all = [], [], []
    
    with torch.no_grad():
        for pids, lang_inputs, acoustic_inputs, labels in loader:
            lang_inputs = tuple(x.to(device) for x in lang_inputs)
            acoustic_inputs = tuple(x.to(device) for x in acoustic_inputs)
            
            preds = model(lang_inputs, acoustic_inputs)
            
            preds_all.append(preds.cpu().numpy())
            labels_all.append(labels.numpy())
            pids_all.extend(pids)

    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    
    if np.any(np.isnan(preds_all)):
        r = np.nan
    else:
        r = pearsonr(labels_all, preds_all)[0] if np.std(preds_all) > 0 else 0.0
        
    # Relative Error
    max_label = np.max(labels_all) if np.max(labels_all) > 0 else 1.0
    re = np.mean(np.abs(preds_all - labels_all) / max_label)
    
    return r, re, pids_all, preds_all, labels_all


# ====================================================================
# MAIN
# ====================================================================

if __name__ == "__main__":
    print("="*60)
    print("MULTIMODAL FUSION TRAINING")
    print("="*60)
    
    # Load data
    pids, lang_bert, lang_sent, lang_lengths, acoustic_seqs, acoustic_lengths, phq_scores = load_and_align_data()
    
    # --- New: Initial Train/Test Split ---
    train_idx, test_idx = train_test_split(
        range(len(pids)), test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    
    # Helper to subset the data lists/arrays
    def subset_data(data_list, indices):
        return [data_list[i] for i in indices]

    train_pids = subset_data(pids, train_idx)
    train_lang_bert = subset_data(lang_bert, train_idx)
    train_lang_sent = subset_data(lang_sent, train_idx)
    train_lang_lengths = subset_data(lang_lengths, train_idx)
    train_acoustic_seqs = subset_data(acoustic_seqs, train_idx)
    train_acoustic_lengths = subset_data(acoustic_lengths, train_idx)
    train_phq_scores = phq_scores[train_idx]
    
    test_pids = subset_data(pids, test_idx)
    test_lang_bert = subset_data(lang_bert, test_idx)
    test_lang_sent = subset_data(lang_sent, test_idx)
    test_lang_lengths = subset_data(lang_lengths, test_idx)
    test_acoustic_seqs = subset_data(acoustic_seqs, test_idx)
    test_acoustic_lengths = subset_data(acoustic_lengths, test_idx)
    test_phq_scores = phq_scores[test_idx]

    # Load pre-trained models
    print("\nLoading pre-trained models...")
    with open(LANGUAGE_HPARAMS, "r") as f:
        lang_hp = json.load(f)
    
    bert_dim = len(lang_bert[0][0])
    
    # Load and apply scaler
    with open(ACOUSTIC_SCALER, "rb") as f:
        scaler = pickle.load(f)
    train_acoustic_seqs = [scaler.transform(seq) for seq in train_acoustic_seqs]
    test_acoustic_seqs = [scaler.transform(seq) for seq in test_acoustic_seqs] # Scale test set
    
    acoustic_dim = acoustic_seqs[0].shape[1]
    
    print(f"✓ Models loaded (lang_hidden={lang_hp['hidden_size']}, acoustic_dim={acoustic_dim})")
    print(f"Data split: Train (for CV)={len(train_idx)}, Test (holdout)={len(test_idx)}")
    
    # Create dataset for CV (Train data only)
    train_cv_dataset = MultimodalDataset(
        train_pids, train_lang_bert, train_lang_sent, train_lang_lengths, 
        train_acoustic_seqs, train_acoustic_lengths, train_phq_scores
    )
    
    # Grid search
    print("\n" + "="*60)
    print("HYPERPARAMETER SEARCH (CV on Train Set)")
    print("="*60)
    
    keys, values = zip(*FUSION_GRID.items())
    combos = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    results = []
    best_r = -1.0
    best_config = None
    
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    start_time = time.time()
    
    for combo_idx, combo in enumerate(combos, 1):
        print(f"\n[{combo_idx}/{len(combos)}] Testing: {combo}")
        
        cv_preds_list, cv_labels_list, cv_pids_list = [], [], [] # For saving CV results
        
        # We need indices relative to the train_cv_dataset
        train_cv_indices = np.arange(len(train_cv_dataset))
        
        for fold, (cv_train_idx, cv_val_idx) in enumerate(kf.split(train_cv_indices), 1):
            
            # Create fresh model architectures
            lang_m = ContextualLSTM(bert_dim, lang_hp["hidden_size"], 
                                   lang_hp["num_layers"], lang_hp["dropout"], lang_hp["fusion"])
            lang_m.load_state_dict(torch.load(LANGUAGE_MODEL, map_location=DEVICE))
            
            acoustic_m = AcousticLSTM(acoustic_dim, 128, 1, 0.3)
            acoustic_m.load_state_dict(torch.load(ACOUSTIC_MODEL, map_location=DEVICE))
            
            model = MultimodalFusion(lang_m, acoustic_m, combo["fusion_hidden"], 
                                     combo["dropout"]).to(DEVICE)
            
            optimizer = torch.optim.Adam(model.parameters(), lr=combo["lr"], 
                                         weight_decay=WEIGHT_DECAY)
            criterion = nn.MSELoss()
            
            # Data loaders
            cv_train_loader = DataLoader(torch.utils.data.Subset(train_cv_dataset, cv_train_idx),
                                         batch_size=combo["batch_size"], shuffle=True, collate_fn=collate_fn)
            cv_val_loader = DataLoader(torch.utils.data.Subset(train_cv_dataset, cv_val_idx),
                                       batch_size=combo["batch_size"], shuffle=False, collate_fn=collate_fn)
            
            # Train with early stopping
            best_re = float("inf")
            patience = 0
            best_state = model.state_dict()
            
            for epoch in range(1, EPOCHS + 1):
                loss = train_epoch(model, cv_train_loader, optimizer, criterion, DEVICE)
                r_val, re_val, _, _, _ = evaluate_with_pids(model, cv_val_loader, DEVICE)
                
                if np.isnan(loss):
                    break
                
                if re_val < best_re:
                    best_re = re_val
                    patience = 0
                    # Save best model state relative to this fold
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= PATIENCE:
                        break
            
            # Evaluate best model on validation fold and collect predictions
            model.load_state_dict(best_state)
            r_test, re_test, pids_val, preds_val, labels_val = evaluate_with_pids(model, cv_val_loader, DEVICE)
            
            print(f"  Fold {fold}: r={r_test:.4f}, RE={re_test:.4f}")
            
            # Store predictions for aggregate CV score and saving
            cv_pids_list.extend(pids_val)
            cv_preds_list.extend(preds_val)
            cv_labels_list.extend(labels_val)
        
        # Aggregate CV performance
        all_preds = np.array(cv_preds_list)
        all_labels = np.array(cv_labels_list)
        
        agg_r = pearsonr(all_labels, all_preds)[0] if np.std(all_preds) > 0 else 0.0
        max_label = np.max(all_labels) if np.max(all_labels) > 0 else 1.0
        agg_re = np.mean(np.abs(all_preds - all_labels) / max_label)
        
        print(f">>> AGGREGATE: r={agg_r:.4f}, RE={agg_re:.4f}")
        
        combo_result = {
            **combo, 
            "agg_r": float(agg_r), 
            "agg_re": float(agg_re),
            "cv_pids": cv_pids_list, # New: Store PIDs and results for saving
            "cv_preds": cv_preds_list,
            "cv_labels": cv_labels_list
        }
        
        if agg_r > best_r:
            best_r = agg_r
            best_config = combo_result # Save the entire result including CV data
        
        results.append(combo_result)
    
    # Save best config and CV results
    print("\n" + "="*60)
    print(f"COMPLETE - Time: {(time.time()-start_time)/60:.1f} min")
    print(f"🏆 BEST CV CONFIG: r={best_r:.4f}")
    
    # Save a clean version of best config (excluding the massive lists)
    best_config_clean = {k: v for k, v in best_config.items() if k not in ("cv_pids", "cv_preds", "cv_labels")}
    print(json.dumps(best_config_clean, indent=2))
    
    with open(os.path.join(FUSION_MODEL_DIR, "best_config.json"), "w") as f:
        json.dump(best_config_clean, f, indent=2)

    # New: Save the CV predictions for analysis
    cv_analysis_df = {
        "PID": best_config["cv_pids"],
        "True_PHQ": best_config["cv_labels"],
        "Predicted_PHQ": best_config["cv_preds"]
    }
    with open(os.path.join(FUSION_MODEL_DIR, "best_cv_predictions.pkl"), "wb") as f:
        pickle.dump(cv_analysis_df, f)
    print(f"Saved CV predictions to: {os.path.join(FUSION_MODEL_DIR, 'best_cv_predictions.pkl')}")

    # --- Retrain Final Model on ALL Training Data ---
    print("\nRetraining final model on ALL **Training** data...")
    lang_final = ContextualLSTM(bert_dim, lang_hp["hidden_size"], lang_hp["num_layers"], 
                                 lang_hp["dropout"], lang_hp["fusion"])
    lang_final.load_state_dict(torch.load(LANGUAGE_MODEL, map_location=DEVICE))
    
    acoustic_final = AcousticLSTM(acoustic_dim, 128, 1, 0.3)
    acoustic_final.load_state_dict(torch.load(ACOUSTIC_MODEL, map_location=DEVICE))
    
    final_model = MultimodalFusion(lang_final, acoustic_final, 
                                   best_config_clean["fusion_hidden"], 
                                   best_config_clean["dropout"]).to(DEVICE)
    
    optimizer = torch.optim.Adam(final_model.parameters(), lr=best_config_clean["lr"], 
                                 weight_decay=WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    # Training loader uses the full training set (the one used for K-Fold)
    loader = DataLoader(train_cv_dataset, batch_size=best_config_clean["batch_size"], 
                        shuffle=True, collate_fn=collate_fn)
    
    best_state_final = final_model.state_dict()
    best_loss = float('inf')
    
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(final_model, loader, optimizer, criterion, DEVICE)
        if loss < best_loss:
            best_loss = loss
            # Saving state based on lowest training loss across epochs (simple approach)
            best_state_final = {k: v.cpu() for k, v in final_model.state_dict().items()}
        
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: loss={loss:.4f}")
    
    final_model.load_state_dict(best_state_final)
    torch.save(final_model.state_dict(), os.path.join(FUSION_MODEL_DIR, "fusion_best.pt"))
    print("\n✅ Final model saved to:", os.path.join(FUSION_MODEL_DIR, "fusion_best.pt"))

    # --- New: Final Evaluation on Holdout Test Set ---
    print("\n" + "="*60)
    print("FINAL EVALUATION ON HOLDOUT TEST SET")
    print("="*60)
    
    # Create test dataset and loader
    test_dataset = MultimodalDataset(
        test_pids, test_lang_bert, test_lang_sent, test_lang_lengths, 
        test_acoustic_seqs, test_acoustic_lengths, test_phq_scores
    )
    test_loader = DataLoader(test_dataset, batch_size=best_config_clean["batch_size"], 
                             shuffle=False, collate_fn=collate_fn)
    
    # Evaluate final model
    r_final, re_final, pids_final, preds_final, labels_final = evaluate_with_pids(final_model, test_loader, DEVICE)

    print(f"**Final Test Set Performance (N={len(test_idx)}):**")
    print(f"  Pearson's r: **{r_final:.4f}**")
    print(f"  Relative Error (RE): **{re_final:.4f}**")
    
    # Save test set predictions for reporting
    test_analysis_df = {
        "PID": pids_final,
        "True_PHQ": labels_final,
        "Predicted_PHQ": preds_final
    }
    with open(os.path.join(FUSION_MODEL_DIR, "final_test_predictions.pkl"), "wb") as f:
        pickle.dump(test_analysis_df, f)
    print(f"Saved final test predictions to: {os.path.join(FUSION_MODEL_DIR, 'final_test_predictions.pkl')}")
    
    print("\n✨ Training and complete evaluation cycle finished.")