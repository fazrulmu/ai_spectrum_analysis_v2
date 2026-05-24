import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset
import argparse
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))
from src.models.pytorch_models import MultimodalCNN

def train_multimodal_torch(ir_path, uv_path, target_label, output_dir=Path("reports/multimodal_analysis_torch"), model_dir=Path("models_torch")):
    print(f"🚀 Starting PyTorch training for target: '{target_label}'")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"ℹ️  Using device: {device}")
    if device.type == 'cpu':
        print("⚠️  Running on CPU. Training might be slower but fully functional.")

    # Load Data
    try:
        ir_df = pd.read_csv(ir_path)
        uv_df = pd.read_csv(uv_path)
    except Exception as e:
        print(f"❌ Error loading datasets: {e}")
        return

    # Merge Logic (Simplified from original)
    if target_label not in ir_df.columns:
        print(f"❌ Error: Target label '{target_label}' not found.")
        return

    ir_features = ir_df.filter(like='bin_')
    uv_features = uv_df.filter(like='bin_')
    
    ir_df_renamed = ir_df.rename(columns={col: f"ir_{col}" for col in ir_features.columns})
    uv_df_renamed = uv_df.rename(columns={col: f"uv_{col}" for col in uv_features.columns})
    
    cols_to_keep_ir = ['molecule_id', target_label] + [f"ir_{col}" for col in ir_features.columns]
    cols_to_keep_uv = ['molecule_id'] + [f"uv_{col}" for col in uv_features.columns]
    
    merged_df = pd.merge(ir_df_renamed[cols_to_keep_ir], uv_df_renamed[cols_to_keep_uv], on='molecule_id', how='inner')
    
    if merged_df.empty:
        print("❌ Error: No common samples found.")
        return

    X_ir = merged_df.filter(like='ir_bin_').values.astype(np.float32)
    X_uv = merged_df.filter(like='uv_bin_').values.astype(np.float32)
    y = merged_df[target_label].values.astype(np.float32)
    
    # Split
    X_ir_train, X_ir_test, X_uv_train, X_uv_test, y_train, y_test = train_test_split(
        X_ir, X_uv, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to Tensors
    train_dataset = TensorDataset(torch.tensor(X_ir_train), torch.tensor(X_uv_train), torch.tensor(y_train).unsqueeze(1))
    test_dataset = TensorDataset(torch.tensor(X_ir_test), torch.tensor(X_uv_test), torch.tensor(y_test).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize Model
    model = MultimodalCNN(ir_length=X_ir.shape[1], uv_length=X_uv.shape[1]).to(device)
    
    # Loss and Optimizer
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()]).to(device) # Simple class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # More stable than BCELoss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training Loop
    epochs = 10 # Reduced for CPU demo
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for ir_batch, uv_batch, y_batch in train_loader:
            ir_batch, uv_batch, y_batch = ir_batch.to(device), uv_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(ir_batch, uv_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")
        
    # Evaluation
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for ir_batch, uv_batch, y_batch in test_loader:
            ir_batch, uv_batch = ir_batch.to(device), uv_batch.to(device)
            outputs = model(ir_batch, uv_batch)
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Save Model
    model_path = model_dir / f"multimodal_model_{target_label}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="alcohol", help="Target functional group")
    args = parser.parse_args()
    
    IR_FILE = Path("data/for_train/universal_training_dataset_IR.csv")
    UV_FILE = Path("data/for_train/universal_training_dataset_UV.csv")
    
    if IR_FILE.exists() and UV_FILE.exists():
        train_multimodal_torch(IR_FILE, UV_FILE, args.target)
    else:
        print("❌ Training data not found. Please generate datasets first.")
