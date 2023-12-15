import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 2, kernel_size=3, stride=2, padding=1),
            nn.AvgPool1d(kernel_size=2, stride=2, padding=0),
            nn.Dropout1d(0.2),
            nn.ReLU(),
            nn.Conv1d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.Dropout1d(0.2),
            nn.ReLU(),
            nn.Conv1d(4, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(1, 3, kernel_size=3, stride=2, padding=1),
            nn.Dropout1d(0.2),
            nn.ReLU(),
            nn.ConvTranspose1d(3, 4, kernel_size=3, stride=2, padding=1),
            nn.Dropout1d(0.2),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 1, kernel_size=4, stride=2, padding=0),
            # nn.Sigmoid()
        )
        
    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        # print(x.shape)
        encoded = self.encoder(x)
        # print(encoded.shape)
        decoded = self.decoder(encoded)
        # print(decoded.shape)
        return decoded
