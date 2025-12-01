#!/usr/bin/env python3
"""
Diagnostic script to compare multimodal fusion predictions with individual model predictions.
"""

import json
import pickle
import numpy as np
from scipy.stats import pearsonr

# Paths
MODELS_DIR = r"C:\Users\jayan\ML_Projects\speech_based_machine_learning_models_of_depression\models"

# Load individual model results
with open(f"{MODELS_DIR}/language_final_test_results.json", "r") as f:
    lang_results = json.load(f)

with open(f"{MODELS_DIR}/acoustic_final_test_results.json", "r") as f:
    acoustic_results = json.load(f)

# Load multimodal results
with open(f"{MODELS_DIR}/multimodal/final_test_predictions.pkl", "rb") as f:
    multimodal_results = pickle.load(f)

# Convert to arrays for analysis
true_scores = np.array(lang_results["true_scores"])
lang_preds = np.array(lang_results["predicted_scores"])
acoustic_preds = np.array(acoustic_results["predicted_scores"])
multimodal_preds = np.array(multimodal_results["Predicted_PHQ"])

print("="*70)
print("DIAGNOSTIC ANALYSIS: Individual Models vs Multimodal Fusion")
print("="*70)
print(f"\nTest Set Size: {len(true_scores)} participants")
print(f"\nPIDs: {lang_results['test_pids']}")

print("\n" + "-"*70)
print("PERFORMANCE COMPARISON")
print("-"*70)

# Language model
lang_r = pearsonr(true_scores, lang_preds)[0]
lang_mae = np.mean(np.abs(true_scores - lang_preds))
print(f"\n📝 Language Model:")
print(f"   Correlation (r): {lang_r:.4f}")
print(f"   MAE: {lang_mae:.4f}")

# Acoustic model
acoustic_r = pearsonr(true_scores, acoustic_preds)[0]
acoustic_mae = np.mean(np.abs(true_scores - acoustic_preds))
print(f"\n🎤 Acoustic Model:")
print(f"   Correlation (r): {acoustic_r:.4f}")
print(f"   MAE: {acoustic_mae:.4f}")

# Multimodal fusion
multimodal_r = pearsonr(true_scores, multimodal_preds)[0]
multimodal_mae = np.mean(np.abs(true_scores - multimodal_preds))
print(f"\n🔗 Multimodal Fusion:")
print(f"   Correlation (r): {multimodal_r:.4f}")
print(f"   MAE: {multimodal_mae:.4f}")

# Simple average baseline
avg_preds = (lang_preds + acoustic_preds) / 2
avg_r = pearsonr(true_scores, avg_preds)[0]
avg_mae = np.mean(np.abs(true_scores - avg_preds))
print(f"\n📊 Simple Average Baseline:")
print(f"   Correlation (r): {avg_r:.4f}")
print(f"   MAE: {avg_mae:.4f}")

print("\n" + "-"*70)
print("DETAILED PREDICTIONS")
print("-"*70)
print(f"\n{'PID':<6} {'True':<6} {'Lang':<8} {'Acoustic':<10} {'Fusion':<8} {'Avg':<8}")
print("-"*70)
for i, pid in enumerate(lang_results["test_pids"]):
    print(f"{pid:<6} {true_scores[i]:<6.1f} {lang_preds[i]:<8.2f} {acoustic_preds[i]:<10.2f} "
          f"{multimodal_preds[i]:<8.2f} {avg_preds[i]:<8.2f}")

print("\n" + "-"*70)
print("PREDICTION ANALYSIS")
print("-"*70)

# Check correlation between modalities
lang_acoustic_corr = pearsonr(lang_preds, acoustic_preds)[0]
print(f"\nCorrelation between Language and Acoustic predictions: {lang_acoustic_corr:.4f}")

lang_fusion_corr = pearsonr(lang_preds, multimodal_preds)[0]
acoustic_fusion_corr = pearsonr(acoustic_preds, multimodal_preds)[0]
print(f"Correlation between Language and Fusion: {lang_fusion_corr:.4f}")
print(f"Correlation between Acoustic and Fusion: {acoustic_fusion_corr:.4f}")

# Check if fusion predictions are reasonable
print(f"\nFusion prediction range: [{multimodal_preds.min():.2f}, {multimodal_preds.max():.2f}]")
print(f"True score range: [{true_scores.min():.2f}, {true_scores.max():.2f}]")
print(f"Language prediction range: [{lang_preds.min():.2f}, {lang_preds.max():.2f}]")
print(f"Acoustic prediction range: [{acoustic_preds.min():.2f}, {acoustic_preds.max():.2f}]")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("\nThe multimodal fusion model performs WORSE than:")
print(f"  - Language model alone (r={lang_r:.4f} vs r={multimodal_r:.4f})")
print(f"  - Simple average of both models (r={avg_r:.4f} vs r={multimodal_r:.4f})")
print("\nThis suggests the fusion model is NOT properly combining the modalities.")
print("The pre-trained models may have been fine-tuned during fusion training,")
print("causing them to lose their original learned representations.")
print("="*70)
