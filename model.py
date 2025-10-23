import torch
import torch.nn as nn

class LSTMBinaryClassifier(nn.Module):
    def __init__(self, num_features, hidden_size, dense_size, num_layers, dropout_prob=0.3):

        super(LSTMBinaryClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0        
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(dense_size, dense_size // 2),
            nn.ReLU(),
            nn.Linear(dense_size // 2, 1)        
        )
    
    def forward(self, x):
        _, (h_n, __) = self.lstm(x)
        h_n = h_n[-1]
        return self.classifier(h_n)

class LSTMRegressor(nn.Module):
    def __init__(self, num_features, hidden_size, dense_size, num_layers, dropout_prob=0.3):
        super(LSTMRegressor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )

        self.regressor = nn.Sequential(
            nn.Linear(hidden_size, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(dense_size, dense_size // 2),
            nn.ReLU(),
            nn.Linear(dense_size // 2, 1) 
        )
    
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden_state = h_n[-1]
        
        # Pass to the regressor head
        return self.regressor(last_hidden_state)
