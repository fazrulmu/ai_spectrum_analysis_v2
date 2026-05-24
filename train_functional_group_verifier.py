import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import joblib
import random
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import interp1d

# --- Configuration ---
CONFIG = {
    'pos_data_path': 'data/for_train/functional_group_spectral_data.jsonl',
    'full_data_path': 'data/for_train/universal_training_dataset_IR.jsonl',
    'struct_conf_path': 'data/for_train/structural_confidence.json',
    'model_save_path': 'saved_models/func_group_verifier.pth',
    'encoder_save_path': 'saved_models/func_group_encoder.joblib',
    'slice_length': 128,
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'device': 'cpu' # Force CPU due to ROCm instability on APU
}

# --- 1. Helper Functions ---
def resample_slice(y_vals, target_length=128):
    if len(y_vals) < 2:
        return np.zeros(target_length)
    x_old = np.linspace(0, 1, len(y_vals))
    x_new = np.linspace(0, 1, target_length)
    f = interp1d(x_old, y_vals, kind='linear', fill_value="extrapolate")
    return f(x_new)

# --- 2. Dataset Class with Negative Sampling ---
class FunctionalGroupDataset(Dataset):
    def __init__(self, pos_data, full_data_map, struct_conf, group_to_id, slice_length=128):
        self.pos_data = pos_data # List of dicts
        self.full_data_map = full_data_map # Dict: cas -> {x: [], y: []}
        self.struct_conf = struct_conf
        self.group_to_id = group_to_id
        self.slice_length = slice_length
        
        # Pre-calculate files that DO NOT have each group (for fast negative sampling)
        self.neg_candidates = {}
        all_cas = list(full_data_map.keys())
        
        print("⚡ Pre-calculating negative candidates...")
        all_groups = list(group_to_id.keys())
        
        for grp in tqdm(all_groups):
            candidates = []
            for cas in all_cas:
                # Check if this CAS has the group
                has_group = False
                if cas in struct_conf:
                    if grp in struct_conf[cas].get('detected_functional_groups', {}):
                        has_group = True
                
                if not has_group:
                    candidates.append(cas)
            self.neg_candidates[grp] = candidates

    def __len__(self):
        return len(self.pos_data) * 2 # We aim for 50/50 balance, so virtual size is double positive

    def __getitem__(self, idx):
        # Determine if Positive or Negative based on index parity or random
        # To ensure we use all positive data, we map idx to positive list
        
        real_idx = idx // 2
        is_positive = (idx % 2 == 0)
        
        pos_record = self.pos_data[real_idx % len(self.pos_data)]
        group_name = pos_record['group_name']
        group_id = self.group_to_id[group_name]
        
        if is_positive:
            # --- POSITIVE SAMPLE ---
            y_vals = np.array(pos_record['y_values'])
            label = 1.0
        else:
            # --- NEGATIVE SAMPLE (On-the-fly) ---
            # 1. Find a CAS that does NOT have this group
            candidates = self.neg_candidates.get(group_name, [])
            if not candidates:
                # Fallback: if no negative candidates (rare), use random noise
                y_vals = np.random.rand(self.slice_length) * 0.1
                label = 0.0
            else:
                neg_cas = random.choice(candidates)
                full_spec = self.full_data_map[neg_cas]
                
                # 2. Slice at the same range
                r_min = pos_record['range_min']
                r_max = pos_record['range_max']
                
                # Full spec X is usually 4000 -> 90 (Descending)
                # We need to find indices where r_min <= x <= r_max
                x_full = np.array(full_spec['x']) # Assuming this is available/passed correctly
                # If x is not available in full_data_map, we might need to rely on grid assumption
                # For now let's assume full_data_map has 'y' which corresponds to 4000->90 grid
                
                # Grid logic (assuming standard IR grid 4000->90 with 3911 points)
                # Step = (4000-90) / 3910 = 1.0
                # Index = (4000 - val) / 1.0
                
                idx_start = int(4000 - r_max)
                idx_end = int(4000 - r_min)
                
                # Clip indices
                idx_start = max(0, min(idx_start, 3910))
                idx_end = max(0, min(idx_end, 3910))
                
                if idx_start >= idx_end:
                    # Swap or fix
                    idx_start, idx_end = idx_end, idx_start
                    
                y_full = np.array(full_spec['y'])
                y_slice = y_full[idx_start:idx_end+1]
                
                if len(y_slice) == 0:
                     y_vals = np.zeros(self.slice_length)
                else:
                     y_vals = y_slice
                
                label = 0.0

        # Resample to fixed length
        y_resampled = resample_slice(y_vals, self.slice_length)
        
        return (
            torch.FloatTensor(y_resampled).unsqueeze(0), # (1, 128)
            torch.LongTensor([group_id]),                # (1)
            torch.FloatTensor([label])                   # (1)
        )

# --- 3. Model Architecture ---
class VerifierModel(nn.Module):
    def __init__(self, num_groups, embedding_dim=16, slice_length=128):
        super(VerifierModel, self).__init__()
        
        # Spectral Branch
        self.spec_conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        ) # Output: (Batch, 64, 1) -> Flatten -> 64
        
        # Embedding Branch
        self.group_embed = nn.Embedding(num_groups, embedding_dim)
        
        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(64 + embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x_spec, grp_id):
        # x_spec: (Batch, 1, 128)
        # grp_id: (Batch, 1)
        
        feat = self.spec_conv(x_spec).view(x_spec.size(0), -1) # (Batch, 64)
        emb = self.group_embed(grp_id).view(grp_id.size(0), -1) # (Batch, 16)
        
        combined = torch.cat([feat, emb], dim=1)
        return self.classifier(combined)

# --- 4. Main Training Flow ---
def train_verifier():
    Path(CONFIG['model_save_path']).parent.mkdir(parents=True, exist_ok=True)
    
    # A. Load Positive Data
    print("🔄 Loading positive samples...")
    pos_data = []
    with open(CONFIG['pos_data_path'], 'r') as f:
        for line in f:
            pos_data.append(json.loads(line))
            
    # B. Load Full Data (for Negatives)
    print("🔄 Loading full dataset (for negative sampling)...")
    full_data_map = {}
    # We need to construct a map: CAS -> Spectrum Y
    # Note: universal_training_dataset_IR.jsonl has 'cas_number' and 'y'
    with open(CONFIG['full_data_path'], 'r') as f:
        for line in f:
            rec = json.loads(line)
            cas = rec['cas_number']
            # Only store if we don't have it (or overwrite, doesn't matter much)
            full_data_map[cas] = {'y': rec['y'], 'x': rec['x']}
            
    # C. Load Structural Confidence
    with open(CONFIG['struct_conf_path'], 'r') as f:
        struct_conf = json.load(f)
        
    # D. Build Group Encoder
    unique_groups = sorted(list(set(d['group_name'] for d in pos_data)))
    group_to_id = {g: i for i, g in enumerate(unique_groups)}
    print(f"🏷️  Groups ({len(unique_groups)}): {unique_groups[:5]}...")
    joblib.dump(group_to_id, CONFIG['encoder_save_path'])
    
    # E. Create Dataset & Loader
    dataset = FunctionalGroupDataset(pos_data, full_data_map, struct_conf, group_to_id, CONFIG['slice_length'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # F. Init Model
    device = torch.device(CONFIG['device'])
    model = VerifierModel(num_groups=len(unique_groups)).to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    print(f"🚀 Starting training on {device}...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        
        for x_spec, grp_id, label in tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False):
            x_spec, grp_id, label = x_spec.to(device), grp_id.to(device), label.to(device)
            
            optimizer.zero_grad()
            output = model(x_spec, grp_id)
            loss = criterion(output, label.float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Loss = {train_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), CONFIG['model_save_path'])
    print("💾 Model saved.")

if __name__ == '__main__':
    train_verifier()
