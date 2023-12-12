import torch
import torch.nn as nn

class mlp_Baseline(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(mlp_Baseline, self).__init__()
        self.fc1 = nn.Linear(input_dim, 600)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(600, 400)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(400, 200)
        self.dropout3 = nn.Dropout(0.1)
        self.fc4 = nn.Linear(200, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

class rnn_Baseline(nn.Module):
    def __init__(self, window_size, output_dim):
        super(rnn_Baseline, self).__init__()
        hidden_dim = 128
        self.window_size = window_size
        self.lstm = nn.LSTM(window_size, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x[:, :-(x.size(1) % self.window_size) or None]
        x = x.view(x.size(0), -1, self.window_size)
        _, (h_n, _) = self.lstm(x)
        h_n = torch.cat((h_n[0], h_n[1]), dim=-1) ## for bidirectional LSTM

        x = self.relu(h_n) 
        x = self.dropout(x)
        x = self.fc(x)
        return x
