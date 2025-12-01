#!/usr/bin/env python3
"""
Apply freezing fixes to train_MultiModal_Fusion.py
This script patches the file to freeze pre-trained models.
"""

import re

# Read the file with proper encoding
with open(r"C:\Users\jayan\ML_Projects\speech_based_machine_learning_models_of_depression\notebooks\train_MultiModal_Fusion.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix 1: Add freezing after loading language model in CV loop
pattern1 = r"(lang_m\.load_state_dict\(torch\.load\(LANGUAGE_MODEL, map_location=DEVICE\)\))\n(\s+)\n(\s+acoustic_m = AcousticLSTM)"
replacement1 = r"\1\n\2# Freeze language model to preserve learned representations\n\2for param in lang_m.parameters():\n\2    param.requires_grad = False\n\2\n\3"
content = re.sub(pattern1, replacement1, content)

# Fix 2: Add freezing after loading acoustic model in CV loop
pattern2 = r"(acoustic_m\.load_state_dict\(torch\.load\(ACOUSTIC_MODEL, map_location=DEVICE\)\))\n(\s+)\n(\s+model = MultimodalFusion)"
replacement2 = r"\1\n\2# Freeze acoustic model to preserve learned representations\n\2for param in acoustic_m.parameters():\n\2    param.requires_grad = False\n\2\n\3"
content = re.sub(pattern2, replacement2, content)

# Fix 3: Change optimizer in CV loop to train only fusion parameters
pattern3 = r"optimizer = torch\.optim\.Adam\(model\.parameters\(\), lr=combo\[\"lr\"\],\s+weight_decay=WEIGHT_DECAY\)"
replacement3 = "# Only train fusion network, not pre-trained models\n            optimizer = torch.optim.Adam(model.fusion.parameters(), lr=combo[\"lr\"], \n                                         weight_decay=WEIGHT_DECAY)"
content = re.sub(pattern3, replacement3, content)

# Fix 4: Add freezing for final model training
pattern4 = r"(lang_final\.load_state_dict\(torch\.load\(LANGUAGE_MODEL, map_location=DEVICE\)\))\n(\s+)\n(\s+acoustic_final = AcousticLSTM)"
replacement4 = r"\1\n\2# Freeze language model to preserve learned representations\n\2for param in lang_final.parameters():\n\2    param.requires_grad = False\n\2\n\3"
content = re.sub(pattern4, replacement4, content)

# Fix 5: Add freezing for final acoustic model
pattern5 = r"(acoustic_final\.load_state_dict\(torch\.load\(ACOUSTIC_MODEL, map_location=DEVICE\)\))\n(\s+)\n(\s+final_model = MultimodalFusion)"
replacement5 = r"\1\n\2# Freeze acoustic model to preserve learned representations\n\2for param in acoustic_final.parameters():\n\2    param.requires_grad = False\n\2\n\3"
content = re.sub(pattern5, replacement5, content)

# Fix 6: Change final optimizer to train only fusion parameters
pattern6 = r"optimizer = torch\.optim\.Adam\(final_model\.parameters\(\), lr=best_config_clean\[\"lr\"\],\s+weight_decay=WEIGHT_DECAY\)"
replacement6 = "# Only train fusion network, not pre-trained models\n    optimizer = torch.optim.Adam(final_model.fusion.parameters(), lr=best_config_clean[\"lr\"], \n                                 weight_decay=WEIGHT_DECAY)"
content = re.sub(pattern6, replacement6, content)

# Write the patched file  
with open(r"C:\Users\jayan\ML_Projects\speech_based_machine_learning_models_of_depression\notebooks\train_MultiModal_Fusion.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Successfully applied all freezing fixes to train_MultiModal_Fusion.py")
print("Changes made:")
print("  1. Freeze language model in CV loop")
print("  2. Freeze acoustic model in CV loop")
print("  3. Change CV optimizer to train only fusion parameters")
print("  4. Freeze language model in final training")
print("  5. Freeze acoustic model in final training")
print("  6. Change final optimizer to train only fusion parameters")
