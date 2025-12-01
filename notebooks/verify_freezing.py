#!/usr/bin/env python3
"""
Verification script to test that freezing pre-trained models works correctly.
This creates a minimal example to demonstrate the difference between frozen and unfrozen models.
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr

print("="*70)
print("VERIFICATION: Freezing Pre-trained Models")
print("="*70)

# Create a simple pre-trained model (simulating language/acoustic models)
class PretrainedModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim)
    
    def forward(self, x):
        return self.fc(x)

# Create a simple fusion model
class FusionModel(nn.Module):
    def __init__(self, pretrained_model, fusion_hidden):
        super().__init__()
        self.pretrained = pretrained_model
        self.fusion = nn.Linear(pretrained_model.fc.out_features, fusion_hidden)
        self.output = nn.Linear(fusion_hidden, 1)
    
    def forward(self, x):
        embed = self.pretrained(x)
        fused = torch.relu(self.fusion(embed))
        return self.output(fused)

# Test 1: WITHOUT freezing
print("\n" + "-"*70)
print("TEST 1: Training WITHOUT freezing pre-trained model")
print("-"*70)

pretrained1 = PretrainedModel(10, 5)
# Save initial weights
initial_weights1 = pretrained1.fc.weight.data.clone()

fusion1 = FusionModel(pretrained1, 3)
optimizer1 = torch.optim.Adam(fusion1.parameters(), lr=0.01)

# Train for a few steps
for i in range(5):
    x = torch.randn(4, 10)
    y = torch.randn(4, 1)
    pred = fusion1(x)
    loss = nn.MSELoss()(pred, y)
    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()

# Check if pre-trained weights changed
weight_diff1 = (pretrained1.fc.weight.data - initial_weights1).abs().sum().item()
print(f"Pre-trained model weight change: {weight_diff1:.6f}")
print(f"Result: Pre-trained weights {'CHANGED' if weight_diff1 > 1e-6 else 'UNCHANGED'} ❌")

# Test 2: WITH freezing
print("\n" + "-"*70)
print("TEST 2: Training WITH freezing pre-trained model")
print("-"*70)

pretrained2 = PretrainedModel(10, 5)
# Save initial weights
initial_weights2 = pretrained2.fc.weight.data.clone()

# FREEZE the pre-trained model
for param in pretrained2.parameters():
    param.requires_grad = False

fusion2 = FusionModel(pretrained2, 3)

# Verify only fusion parameters are trainable
trainable_params = sum(p.numel() for p in fusion2.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in fusion2.parameters())
print(f"Trainable parameters: {trainable_params}/{total_params}")

# Use only fusion parameters in optimizer (both fusion and output layers)
fusion_params = list(fusion2.fusion.parameters()) + list(fusion2.output.parameters())
optimizer2 = torch.optim.Adam(fusion_params, lr=0.01)

# Capture initial fusion weights BEFORE training
initial_fusion_weights = fusion2.fusion.weight.data.clone()
initial_output_weights = fusion2.output.weight.data.clone()

# Train for a few steps
for i in range(5):
    x = torch.randn(4, 10)
    y = torch.randn(4, 1)
    pred = fusion2(x)
    loss = nn.MSELoss()(pred, y)
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()

# Check if pre-trained weights changed
weight_diff2 = (pretrained2.fc.weight.data - initial_weights2).abs().sum().item()
print(f"Pre-trained model weight change: {weight_diff2:.6f}")
print(f"Result: Pre-trained weights {'CHANGED' if weight_diff2 > 1e-6 else 'UNCHANGED'} ✓")

# Test 3: Verify fusion still learns
print("\n" + "-"*70)
print("TEST 3: Verify fusion layer still learns")
print("-"*70)

# Calculate weight changes
fusion_weight_diff = (fusion2.fusion.weight.data - initial_fusion_weights).abs().sum().item()
output_weight_diff = (fusion2.output.weight.data - initial_output_weights).abs().sum().item()
total_fusion_diff = fusion_weight_diff + output_weight_diff
print(f"Fusion layer weight change: {fusion_weight_diff:.6f}")
print(f"Output layer weight change: {output_weight_diff:.6f}")
print(f"Total trainable weight change: {total_fusion_diff:.6f}")
print(f"Result: Fusion weights {'CHANGED' if total_fusion_diff > 1e-6 else 'UNCHANGED'} ✓")

# Summary
print("\n" + "="*70)
print("VERIFICATION RESULTS")
print("="*70)

if weight_diff1 > 1e-6 and weight_diff2 < 1e-6 and total_fusion_diff > 1e-6:
    print("\n✅ SUCCESS: Freezing works correctly!")
    print("   - Without freezing: pre-trained model changes (bad)")
    print("   - With freezing: pre-trained model preserved (good)")
    print("   - With freezing: fusion layer still learns (good)")
    print("\n✅ Safe to apply freezing to train_MultiModal_Fusion.py")
else:
    print("\n❌ FAILURE: Something unexpected happened")
    print(f"   - Pre-trained change without freeze: {weight_diff1:.6f}")
    print(f"   - Pre-trained change with freeze: {weight_diff2:.6f}")
    print(f"   - Fusion layer change: {total_fusion_diff:.6f}")

print("="*70)
