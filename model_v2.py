
import math
import torch
import torch.nn as nn


# 1. Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=64, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            max_len, dtype=torch.float32
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer(
            "pe", pe.unsqueeze(0)
        )

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


# 2. Attention Pooling
class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.attention = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: [batch, 168, 64]

        scores = self.attention(x)

        weights = torch.softmax(
            scores, dim=1
        )

        context = torch.sum(
            x * weights, dim=1
        )

        return context


# 3. Transformer V2
class LoadTransformerV2(nn.Module):
    def __init__(
        self,
        input_dim=5,
        d_model=64,
        nhead=4,
        num_layers=2,
        pred_len=24
    ):
        super().__init__()

        self.embedding = nn.Linear(
            input_dim, d_model
        )

        self.pos_encoder = PositionalEncoding(
            d_model
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=0.1,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.pooling = AttentionPooling(d_model)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, pred_len)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)

        context = self.pooling(x)

        output = self.fc(context)

        return output


# 4. Test model
if __name__ == "__main__":
    model = LoadTransformerV2()

    x = torch.randn(32, 168, 5)

    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)

    print(
        "Total parameters:",
        sum(p.numel() for p in model.parameters())
    )
