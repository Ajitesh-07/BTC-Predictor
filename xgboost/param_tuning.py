import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import optuna
import optuna.visualization as vis
from sklearn.model_selection import train_test_split

try:
    df = pd.read_csv('features.csv', index_col=0)
except FileNotFoundError:
    print("Error: 'features.csv' not found.")
    exit()

target = 'target_cls'
features = [col for col in df.columns if col not in ['target_cls', 'target_reg', 'Close']]
X = df[features]
y = df[target]

split_percentage = 0.8
split_index = int(len(df) * split_percentage)

X_train_full = X.iloc[:split_index]
y_train_full = y.iloc[:split_index]

X_test_final = X.iloc[split_index:]
y_test_final = y.iloc[split_index:]

print(f"Total samples: {len(df)}")
print(f"Full Training samples: {len(X_train_full)}")
print(f"Final Test samples: {len(X_test_final)}")
print("-" * 30)

def objective(trial):
    
    X_train_inner, X_val_inner, y_train_inner, y_val_inner = train_test_split(
        X_train_full, y_train_full, test_size=0.1, shuffle=False 
    )

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42,
        
        'device': 'cuda',
        
        'n_estimators': trial.suggest_int('n_estimators', 200, 1500), # Wider range
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
    }

    model = xgb.XGBClassifier(**params)
    
    model.fit(
        X_train_inner, 
        y_train_inner,
        eval_set=[(X_val_inner, y_val_inner)],
        
        verbose=False
    )
    
    preds = model.predict(X_val_inner)
    
    accuracy = accuracy_score(y_val_inner, preds)
    return accuracy


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

print("Optuna study finished.")
print(f"Best trial Accuracy: {study.best_value:.4f}") 
print("Best parameters found:")
print(study.best_params)

print("\nShowing optimization plots...")
vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()