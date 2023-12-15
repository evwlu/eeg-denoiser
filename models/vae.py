import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, feature_size=476, latent_size=2):
        super(VAE, self).__init__()
        
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
        
        # Latent space layers
        self.fc_mean = nn.Linear(feature_size * 2, latent_size)
        self.fc_logvar = nn.Linear(feature_size * 2, latent_size)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(2, 4, kernel_size=3, stride=2, padding=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.ConvTranspose1d(4, 2, kernel_size=3, stride=1, padding=1),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.ConvTranspose1d(2, 1, kernel_size=2, padding=0)
        )
        
    def encode(self, x):
        x = torch.unsqueeze(x, dim=1)
        encoded = self.encoder(x)
        encoded = encoded.view(encoded.size(0), -1)
        mean = self.fc_mean(encoded)
        logvar = self.fc_logvar(encoded)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def decode(self, z):
        z = z.view(z.size(0), -1, 1)
        decoded = self.decoder(z)
        return decoded
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        decoded = self.decode(z)
        return decoded
