import torch
from torch import nn


class MLNNWithAttention(nn.Module):
    def __init__(self,
                 n_hidden: int    = 64,
                 n_heads: int     = 4,
                 dropout: float   = 0.1,
                 attn_dropout: float = 0.1):
        super().__init__()
        # ---- MLP trunk ----
        self.fc1  = nn.Linear(1, n_hidden)
        self.fc2  = nn.Linear(n_hidden, n_hidden)
        self.relu = nn.ReLU()
        # ---- Self-attention block ----
        # PyTorch expects (S, B, E), so we'll treat S=N, B=1
        self.self_attn = nn.MultiheadAttention(
            embed_dim   = n_hidden,
            num_heads   = n_heads,
            dropout     = attn_dropout,
            batch_first = False  # returns (S, B, E)
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm    = nn.LayerNorm(n_hidden)
        # ---- Final read-out ----
        self.out = nn.Linear(n_hidden, 3)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (N,1) normalized time series
        returns: (3,) raw scores for (t_c, m, ω)
        """
        # 1) MLP per-timepoint
        #    h: (N, n_hidden)
        h = self.relu(self.fc1(t))
        h = self.relu(self.fc2(h))

        # 2) Prepare for attention: (S, B, E) = (N,1,n_hidden)
        S, _ = h.shape
        h_attn = h.unsqueeze(1)           # (N, 1, n_hidden)
        # self_attn uses same tensor for query, key, value
        attn_out, _ = self.self_attn(
            query = h_attn,
            key   = h_attn,
            value = h_attn,
            need_weights = False
        )
        # Residual + norm
        h = self.attn_norm(h_attn + self.attn_dropout(attn_out))
        # → still (N,1,n_hidden)

        # 3) project back to (N, n_hidden)
        h = h.squeeze(1)

        # 4) final linear & mean pooling
        raw = self.out(h)         # (N, 3)
        return raw.mean(dim=0)    # (3,)


class CNNWithAttention(nn.Module):
    def __init__(self,
                 channels: int    = 16,
                 kernel_size: int = 3,
                 num_conv_layers: int = 2,
                 n_heads: int     = 4,
                 dropout: float   = 0.1,
                 attn_dropout: float = 0.1):
        super().__init__()
        # ---- Trunk CNN1D ----
        conv_layers = []
        in_ch = 1
        for _ in range(num_conv_layers):
            conv_layers += [
                nn.Conv1d(in_ch, channels, kernel_size, padding=kernel_size//2),
                nn.ReLU()
            ]
            in_ch = channels
        self.conv = nn.Sequential(*conv_layers)

        # ---- Self-attention block ----
        # embed_dim = channels, batch_first=False => (seq_len, batch, embed_dim)
        self.self_attn    = nn.MultiheadAttention(
            embed_dim   = channels,
            num_heads   = n_heads,
            dropout     = attn_dropout,
            batch_first = False
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm    = nn.LayerNorm(channels)

        # ---- Final read-out ----
        self.out = nn.Linear(channels, 3)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (N,1) normalized time series
        returns: (3,) raw scores for (t_c, m, ω)
        """
        # 1) CNN trunk
        #    t → (1,1,N) → conv → (1,channels,N)
        x = t.unsqueeze(0).permute(0,2,1)
        Fmaps = self.conv(x)              # (1, channels, N)

        # 2) prépa pour l'attention : (seq_len, batch, embed_dim)
        #    → on veut H: (N, channels)
        H = Fmaps.squeeze(0).permute(1,0) # (N, channels)
        H_b = H.unsqueeze(1)              # (N, 1, channels)

        # 3) Self-attention (Q=K=V=H_b)
        attn_out, _ = self.self_attn(
            query       = H_b,
            key         = H_b,
            value       = H_b,
            need_weights= False
        )
        # 4) Résidu + Dropout + Norm
        H2 = self.attn_norm(H_b + self.attn_dropout(attn_out))  # (N,1,channels)

        # 5) Retransform to (N, channels)
        H2 = H2.squeeze(1)

        # 6) Projection finale + moyenne
        raw = self.out(H2)             # (N, 3)
        return raw.mean(dim=0)         # (3,)


class RNNWithAttention(nn.Module):
    """
    RNN (LSTM) + self-attention + read-out pour estimer (t_c, m, ω).
    Input: t of shape (N,1)
    Output: raw scores of shape (3,)
    """
    def __init__(self,
                 hidden_size: int = 16,
                 num_layers: int = 1,
                 n_heads: int = 4,
                 dropout: float = 0.1,
                 attn_dropout: float = 0.1):
        super().__init__()
        # ---- Trunk RNN ----
        self.rnn = nn.LSTM(input_size=1,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           batch_first=True)
        # ---- Self-attention block ----
        # embed_dim = hidden_size, expects (seq_len, batch, embed_dim)
        self.self_attn    = nn.MultiheadAttention(
            embed_dim   = hidden_size,
            num_heads   = n_heads,
            dropout     = attn_dropout,
            batch_first = False
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm    = nn.LayerNorm(hidden_size)
        # ---- Final read-out ----
        self.out = nn.Linear(hidden_size, 3)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (N,1) → add batch dim → (1, N, 1)
        x, _ = self.rnn(t.unsqueeze(0))      # x: (1, N, hidden_size)
        H     = x.squeeze(0)                 # H: (N, hidden_size)

        # Prepare for attention: (seq_len, batch, embed_dim)
        H_b = H.unsqueeze(1)                 # → (N, 1, hidden_size)
        attn_out, _ = self.self_attn(
            query        = H_b,
            key          = H_b,
            value        = H_b,
            need_weights = False
        )
        # Residual + dropout + norm
        H2 = self.attn_norm(H_b + self.attn_dropout(attn_out))  # (N,1,hidden_size)
        H2 = H2.squeeze(1)                                      # → (N, hidden_size)

        # Project each time-step to 3 scores, then average
        raw = self.out(H2)              # (N, 3)
        return raw.mean(dim=0)          # → (3,)


class NLCNNWithAttention(nn.Module):
    """
    Nonlinear CNN1D + self-attention for LPPLS parameter estimation.
    Input: t of shape (N,1)
    Output: raw scores of shape (3,)
    """
    def __init__(self,
                 channels: int    = 16,
                 kernel_size: int = 3,
                 num_layers: int  = 2,
                 eps: float       = 1e-6,
                 n_heads: int     = 4,
                 attn_dropout: float = 0.1,
                 dropout: float      = 0.1):
        super().__init__()
        self.eps = eps

        # ---- Nonlinear CNN trunk ----
        self.convs       = nn.ModuleList()
        self.exp_weights = nn.ParameterList()
        in_ch = 1
        for _ in range(num_layers):
            conv = nn.Conv1d(in_ch, channels, kernel_size, padding=kernel_size//2)
            exp_w = nn.Parameter(torch.ones_like(conv.weight))
            self.convs.append(conv)
            self.exp_weights.append(exp_w)
            in_ch = channels

        # ---- Self-attention block ----
        # embed_dim = channels, expects (seq_len, batch, embed_dim)
        self.self_attn    = nn.MultiheadAttention(
            embed_dim   = channels,
            num_heads   = n_heads,
            dropout     = attn_dropout,
            batch_first = False
        )
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm    = nn.LayerNorm(channels)

        # ---- Final read-out ----
        self.out = nn.Linear(channels, 3)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (N,1) → x: (1,1,N)
        x = t.unsqueeze(0).permute(0,2,1)

        # 1) apply each nonlinear conv layer
        for conv, exp_w in zip(self.convs, self.exp_weights):
            # weight the kernel by exp(exp_w)
            weighted_kernel = conv.weight * torch.exp(exp_w)
            x = F.conv1d(x,
                         weighted_kernel,
                         bias=conv.bias,
                         padding=conv.padding[0])
            x = F.relu(x)  # (1, channels, N)

        # 2) prepare sequence for attention: H: (N, channels)
        H   = x.squeeze(0).permute(1,0)   # (N, channels)
        H_b = H.unsqueeze(1)              # (N, 1, channels)

        # 3) self-attention (Q=K=V=H_b)
        attn_out, _ = self.self_attn(
            query        = H_b,
            key          = H_b,
            value        = H_b,
            need_weights = False
        )

        # 4) residual + dropout + norm → H2: (N, channels)
        H2 = self.attn_norm(H_b + self.attn_dropout(attn_out))
        H2 = H2.squeeze(1)  # (N, channels)

        # 5) project each time-step to 3 scores, then average
        raw = self.out(H2)       # (N, 3)
        return raw.mean(dim=0)   # (3,)