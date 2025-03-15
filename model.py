import torch
import torch.nn as nn

class ChestXrayClassifier(nn.Module):
    def __init__(self, input_dim=1376, hidden_dims=[512, 384, 256], output_dim=14):
        super(ChestXrayClassifier, self).__init__()
        
        # Input normalization layer (learnable)
        self.batch_norm_input = nn.BatchNorm1d(input_dim)
        
        # Main network layers
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.bn1 = nn.BatchNorm1d(hidden_dims[0])
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.bn2 = nn.BatchNorm1d(hidden_dims[1])
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.bn3 = nn.BatchNorm1d(hidden_dims[2])
        self.dropout3 = nn.Dropout(0.3)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[2], output_dim)
    
    def forward(self, x):
        # Input normalization
        x = self.batch_norm_input(x)
        
        # First layer
        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x)
        x = self.dropout1(x)
        
        # Second layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)
        
        # Third layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x