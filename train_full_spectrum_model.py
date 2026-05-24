import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import joblib
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import os

# --- Configuration ---
CONFIG = {
    'data_path': 'data/for_train/universal_training_dataset_IR.jsonl',
    'model_save_path': 'saved_models/full_spectrum_model.pth',
    'encoder_save_path': 'saved_models/full_spectrum_encoder.joblib',
    'input_length': 3911,
    'batch_size': 32,
    'epochs': 20,
    'learning_rate': 0.001,
    'device': 'cpu' # Force CPU due to ROCm instability on APU
}

# --- 1. Augmentation Class ---
class SpectrumAugmenter:
    def __init__(self, shift_limit=5, noise_scale=0.005, scale_limit=0.1):
        self.shift_limit = shift_limit
        self.noise_scale = noise_scale
        self.scale_limit = scale_limit

    def __call__(self, x):
        # x is numpy array (Length,)
        
        # 1. Random Shift (Roll)
        shift = np.random.randint(-self.shift_limit, self.shift_limit + 1)
        if shift != 0:
            x = np.roll(x, shift)
            # Fix edges (simple repeat or zero) - here we just let it roll or zero out
            if shift > 0: x[:shift] = 0
            else: x[shift:] = 0
            
        # 2. Random Noise
        noise = np.random.normal(0, self.noise_scale, x.shape)
        x = x + noise
        
        # 3. Random Scaling
        scale = 1.0 + np.random.uniform(-self.scale_limit, self.scale_limit)
        x = x * scale
        
        return x

# --- 2. Dataset Class ---
class SpectrumDataset(Dataset):
    def __init__(self, x_data, y_data, augment=False):
        self.x = x_data # Keep as numpy for on-the-fly aug
        self.y = torch.FloatTensor(y_data)
        self.augment = augment
        self.augmenter = SpectrumAugmenter() if augment else None
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        x_sample = self.x[idx].copy() # Copy to avoid modifying original
        
        if self.augment:
            x_sample = self.augmenter(x_sample)
            
        # Add channel dimension: (Length) -> (1, Length)
        return torch.FloatTensor(x_sample).unsqueeze(0), self.y[idx]

# --- 2. Model Architecture (1D CNN) ---
class FullSpectrumCNN(nn.Module):
    def __init__(self, num_classes, input_length=3911):
        super(FullSpectrumCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 2
            nn.Conv1d(32, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            # Block 3
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1) # Global Average Pooling
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
            # No Sigmoid here, we use BCEWithLogitsLoss
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 3. Data Loading & Preprocessing ---
def load_data(path, input_length):
    print(f"🔄 Loading data from {path}...")
    spectra = []
    labels = []
    
    with open(path, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                x = np.array(record['y']) # 'y' in JSON is absorbance (the signal)
                lbls = record['labels']
                
                # Preprocessing: Pad or Truncate to fixed length
                if len(x) > input_length:
                    x = x[:input_length]
                elif len(x) < input_length:
                    pad_width = input_length - len(x)
                    x = np.pad(x, (0, pad_width), mode='constant')
                
                spectra.append(x)
                labels.append(lbls)
            except Exception as e:
                continue
                
    return np.array(spectra), labels

# --- 4. Training Loop ---
def train_model():
    Path(CONFIG['model_save_path']).parent.mkdir(parents=True, exist_ok=True)
    
    # Load Data
    X, y_raw = load_data(CONFIG['data_path'], CONFIG['input_length'])
    
    # Encode Labels
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(y_raw)
    classes = mlb.classes_
    print(f"✅ Loaded {len(X)} samples.")
    print(f"🏷️  Classes ({len(classes)}): {classes}")
    
    # Save Encoder
    joblib.dump(mlb, CONFIG['encoder_save_path'])
    
    # Split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Dataloaders
    train_ds = SpectrumDataset(X_train, y_train, augment=True) # Enable Augmentation
    val_ds = SpectrumDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'])
    
    # Init Model
    device = torch.device(CONFIG['device'])
    model = FullSpectrumCNN(num_classes=len(classes), input_length=CONFIG['input_length']).to(device)
    
    # Calculate Class Weights (Pos Weight)
    # pos_weight = (num_neg / num_pos)
    y_tensor = torch.FloatTensor(y)
    num_pos = y_tensor.sum(dim=0)
    num_neg = len(y) - num_pos
    pos_weights = (num_neg / (num_pos + 1e-5)) # Add epsilon to avoid div by zero
    pos_weights = pos_weights.to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    print(f"🚀 Starting training on {device}...")
    
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}", leave=False):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss/len(train_loader):.4f}, Val Loss = {val_loss/len(val_loader):.4f}")
        
    # Save Model
    torch.save(model.state_dict(), CONFIG['model_save_path'])
    print(f"💾 Model saved to {CONFIG['model_save_path']}")

if __name__ == '__main__':
    train_model()
