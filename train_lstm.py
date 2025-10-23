import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import joblib
from model import LSTMBinaryClassifier, LSTMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = X
        self.y = y
        self.seq_len = seq_len

    def __len__(self):
        return self.X.shape[0] - self.seq_len + 1

    def __getitem__(self, index):
        X_seq = self.X[index : index + self.seq_len]
        y_target = self.y[index + self.seq_len - 1]
        
        return X_seq, y_target

SEQ_LEN = 1
BATCH_SIZE = 128
TEST_SPLIT = 0.2 

def load_and_preprocess_data(csv_path, target_col):
    df = pd.read_csv(csv_path, index_col=0)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not in CSV.")
        
    features = [col for col in df.columns if col not in ['target_cls', 'target_reg', 'Close']]
    
    X = df[features]
    y = df[target_col]

    split_index = int(len(df) * (1 - TEST_SPLIT))
    
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, "scaler.pkl")


    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
    
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)
    
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

if __name__ == '__main__':
    TYPE = 'reg'

    NUM_FEATURES = 19
    LR = 1e-2
    EPOCHS = 10

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using Device {device}")

    X_train, y_train, X_test, y_test = load_and_preprocess_data(
        csv_path='features.csv',
        target_col='target_cls' if TYPE == 'cls' else 'target_reg'
    )

    train_dataset = TimeSeriesDataset(X_train, y_train, seq_len=SEQ_LEN)
    test_dataset = TimeSeriesDataset(X_test, y_test, seq_len=SEQ_LEN)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False
    )


    if TYPE == 'reg':
        model = LSTMRegressor(NUM_FEATURES, 16, 32, 2)
        criterion = nn.MSELoss()
        best_metric = float('inf') # Best (lowest) validation loss

        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=LR)
        print(f"Model: {model.__class__.__name__}")
        print(f"Total Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

        # --- Training & Evaluation Functions ---

        def train_one_epoch():
            model.train()
            running_loss = 0.0
            for features, targets in tqdm(train_loader, desc="Training"):
                features, targets = features.to(device), targets.to(device)
                
                outputs = model(features).squeeze()
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            return running_loss / len(train_loader)

        def evaluate():
            model.eval()
            running_loss = 0.0
            all_preds, all_targets = [], []
            
            with torch.no_grad():
                for features, targets in tqdm(test_loader, desc="Evaluating"):
                    features, targets = features.to(device), targets.to(device)
                    
                    outputs = model(features).squeeze()
                    loss = criterion(outputs, targets)
                    running_loss += loss.item()

                    preds = outputs

                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
            avg_loss = running_loss / len(test_loader)

            mse = mean_squared_error(all_targets, all_preds)
            return avg_loss, mse

        # --- Main Training Loop ---
        print("\n--- Starting Training ---")
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch()
            
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.6f}")
            
            val_loss, val_mse = evaluate()
            print(f"  Val Loss (MSE): {val_loss:.6f}")
            
            if val_loss < best_metric:
                print(f"  New best validation loss! Saving model...")
                torch.save(model.state_dict(), 'best_model_reg.pth')
                best_metric = val_loss

            print("-" * 30)

        print("--- Training Complete ---")

    elif TYPE == 'cls':

        model = LSTMBinaryClassifier(NUM_FEATURES, 128, 256, 4)
        model.to(device=device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LR)

        print(f"Total Params: {sum([p.numel() for p in model.parameters() if p.requires_grad])}")

        def train_one_epoch():
            model.train()
            running_loss = 0.0
            
            for features, targets in tqdm(train_loader, desc="Training"):
                features = features.to(device)
                targets = targets.to(device)
                
                outputs = model(features).squeeze() 
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
            return running_loss / len(train_loader)

        def evaluate():
            model.eval()
            running_loss = 0.0
            all_preds = []
            all_targets = []
            
            with torch.no_grad():
                for features, targets in tqdm(test_loader, desc="Evaluating"):
                    features = features.to(device)
                    targets = targets.to(device)
                    
                    outputs = model(features).squeeze()
                    loss = criterion(outputs, targets)
                    
                    running_loss += loss.item()
                    
                    preds = (torch.sigmoid(outputs) > 0.5).float()
                    
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())
                    
            avg_loss = running_loss / len(test_loader)
            accuracy = accuracy_score(all_targets, all_preds)
            precision = precision_score(all_targets, all_preds, zero_division=0)
            recall = recall_score(all_targets, all_preds, zero_division=0)
            f1 = f1_score(all_targets, all_preds, zero_division=0)
            
            return avg_loss, accuracy, precision, recall, f1

        print("\n--- Starting Training ---")
        best_acc = 0.5232
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch()
            val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate()

            if (val_acc > best_acc):
                torch.save(model.state_dict(), 'best_model.pth')
                best_acc = val_acc
            
            print(f"\nEpoch {epoch+1}/{EPOCHS}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_acc*100:.2f}%")
            print(f"  Precision:  {val_prec:.4f}")
            print(f"  Recall:     {val_rec:.4f}")
            print(f"  F1-Score:   {val_f1:.4f}")
            print("-" * 30)

        print("--- Training Complete ---")