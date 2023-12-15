import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=3, padding=1),
            nn.AvgPool1d(kernel_size=2, padding=0),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv1d(2, 4, kernel_size=3, padding=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Conv1d(4, 4, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 2, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.ConvTranspose1d(2, 1, kernel_size=2, padding=0)
        )
        
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded