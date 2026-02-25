import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        n_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        out_size=1,
        use_attention=True,
        deep_head=False,
    ):
        super(LSTMRegressor, self).__init__()
        self.hidden_size = hidden_size
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size

        if self.use_attention:
            self.layer_norm1 = nn.LayerNorm(lstm_out_size)
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_out_size,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.layer_norm2 = nn.LayerNorm(lstm_out_size)

        # Prediction Head: hidden_size 와 deep_head 에 따라 구조 결정
        # deep_head=True (flow, toc, ss): 4-layer (h → h/2 → h/4 → h/8 → out)
        # deep_head=False + hidden>=256  (tn, tp, flux, ph): 3-layer (h → h/2 → h/4 → out)
        # hidden<256: 2-layer (h → h/2 → out)
        if hidden_size >= 256 and deep_head:
            self.head = nn.Sequential(
                nn.Linear(lstm_out_size, lstm_out_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_size // 2, lstm_out_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_size // 4, lstm_out_size // 8),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_size // 8, out_size),
            )
        elif hidden_size >= 256:
            self.head = nn.Sequential(
                nn.Linear(lstm_out_size, lstm_out_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_size // 2, lstm_out_size // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_size // 4, out_size),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(lstm_out_size, lstm_out_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(lstm_out_size // 2, out_size),
            )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        if self.use_attention:
            lstm_out_normed = self.layer_norm1(lstm_out)
            attn_out, _ = self.attention(lstm_out_normed, lstm_out_normed, lstm_out_normed)
            attn_out = attn_out + lstm_out
            attn_out = self.layer_norm2(attn_out)
            last = attn_out[:, -1, :]
        else:
            last = lstm_out[:, -1, :]

        yhat = self.head(last)
        return yhat


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x):
        self.mean_ = x.mean(axis=0, keepdims=True)
        self.std_ = x.std(axis=0, keepdims=True) + 1e-8
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x):
        return x * self.std_ + self.mean_
