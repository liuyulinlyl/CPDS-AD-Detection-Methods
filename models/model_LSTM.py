import torch
import torch.nn as nn

class LSTM_Block(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.proj = nn.Linear(lstm_out_dim, input_dim)

    def forward(self, x):
        """
        x: (batch_size, window_size, features)
        return: (batch_size, window_size, features)
        """
        out, _ = self.lstm(x)          # (B, T, H)
        out = self.proj(out)           # (B, T, F)
        return out
    
batch_size = 8
window_size = 50
features = 32

x = torch.randn(batch_size, window_size, features)

lstm_block = LSTM_Block(
    input_dim=features,
    hidden_dim=64,
    num_layers=2,
    dropout=0.1,
)

y = lstm_block(x)
print(x.shape, y.shape)
# torch.Size([8, 50, 32]) torch.Size([8, 50, 32])

