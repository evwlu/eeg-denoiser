
import torch.nn as nn

class mlp_Baseline(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(mlp_Baseline, self).__init__()
        self.fc1 = nn.Linear(input_dim, 600)
        self.fc2 = nn.Linear(600, 400)
        self.fc3 = nn.Linear(400, 200)
        self.fc4 = nn.Linear(200, output_dim)
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = (self.fc4(x))
        # x = self.softmax(x)
        return x

class rnn_Baseline(nn.Module):
    def __init__(self, window_size, output_dim):
        super(rnn_Baseline, self).__init__()
        self.window_size = window_size
        self.lstm = nn.LSTM(window_size, 128, batch_first=True)
        self.fc = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x[:, :-(x.size(1) % self.window_size) or None]
        x = x.view(x.size(0), -1, self.window_size)
        _, (h_n, _) = self.lstm(x)
        x = self.relu(h_n[-1]) 
        x = self.fc(x)
        return x


