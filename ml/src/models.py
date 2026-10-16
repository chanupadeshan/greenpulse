# Define Dense Autoencoder and LSTM Autoencoder models for anomaly detection.
# Both models learn to reconstruct sensor readings and identify unusual patterns.


from torch import nn
class DenseAutoencoder(nn.Module):
    """
    Dence autoencoder for individual sensor readings.
    """

    def __init__(self, input_dim, hidden_dim=32, latent_dim=8, dropout=0.1):
        super().__init__()

        # encoder part
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )

        # decoder part
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    ## define feedforward layer
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class LSTMAutoencoder(nn.Module):
    """
    LSTM autoencoder for sequences of sensor readings.
    """

    def __init__(self, input_dim, hidden_dim=64, latent_dim=16):
        super().__init__()

        # encoder part
        self.encoder = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        # decoder part
        self.decoder = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    ## define feedforward layer
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        latent = self.to_latent(hidden[-1])
        decoded_seed = self.from_latent(latent)
        # Repeat the summary for every hour so the decoder produces a reconstruction of equal length.
        repeated = decoded_seed.unsqueeze(1).repeat(1, x.size(1), 1)
        decoded, _ = self.decoder(repeated)
        return self.output(decoded)
