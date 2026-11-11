"""Time attention and horizon-aligned variable attention for compact ATM."""
import torch
from torch import nn
from torch.nn import functional as F


class TokenAttention(nn.Module):
    def __init__(self, dim, heads, dropout, rope=False):
        super().__init__()
        if heads < 1 or dim % heads:
            raise ValueError('d_model must be divisible by n_heads')
        self.heads, self.head_dim = heads, dim // heads
        if rope and self.head_dim % 2:
            raise ValueError('RoPE requires an even attention head dimension')
        self.rope, self.dropout = rope, dropout
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)
        self.register_buffer('inv_freq', 10000.0 ** (
            -torch.arange(0, self.head_dim, 2).float() / self.head_dim), persistent=False)

    def rotate(self, x, positions):
        angles = positions.float()[:, None, :, None] * self.inv_freq.float()
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even*cos-odd*sin, even*sin+odd*cos), -1).flatten(-2)

    def forward(self, x, valid=None, positions=None, identity=None):
        batch, length, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        if identity is not None:
            q, k = q + identity[0], k + identity[1]
        q, k, v = [t.reshape(batch, length, self.heads, self.head_dim).transpose(1, 2)
                   for t in (q, k, v)]
        if self.rope:
            q, k = self.rotate(q, positions), self.rotate(k, positions)
        # SDPA boolean mask: True means a key is allowed to participate.
        mask = None if valid is None else valid[:, None, None, :]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(batch, length, dim))


class ForecastTokenBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        hidden = int(config.inter_dim)
        if hidden < 1:
            raise ValueError('inter_dim must be positive')
        self.time_norm = nn.LayerNorm(d)
        self.time_attn = TokenAttention(d, config.n_heads, config.dropout, bool(config.use_rope))
        self.channel_enabled = config.channel_attention
        if self.channel_enabled:
            self.channel_norm = nn.LayerNorm(d)
            self.channel_attn = TokenAttention(d, config.n_heads, config.dropout)
            self.identity_q = nn.Linear(d, d, bias=False)
            self.identity_k = nn.Linear(d, d, bias=False)
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                 nn.Dropout(config.dropout), nn.Linear(hidden, d))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, valid, positions, n_forecast, channel_identity):
        b, c, length, d = x.shape
        mask = valid.unsqueeze(-1)
        x = x.masked_fill(~mask, 0)
        u = self.time_norm(x).reshape(b*c, length, d)
        x = x + self.dropout(self.time_attn(u, valid.reshape(b*c, length),
            positions.reshape(b*c, length))).reshape(b, c, length, d)
        x = x.masked_fill(~mask, 0)
        if self.channel_enabled:
            forecast = x[:, :, -n_forecast:]
            u = self.channel_norm(forecast).permute(0, 2, 1, 3).reshape(b*n_forecast, c, d)
            identity = (self.identity_q(channel_identity), self.identity_k(channel_identity))
            update = self.channel_attn(u, identity=identity)
            update = update.reshape(b, n_forecast, c, d).permute(0, 2, 1, 3)
            forecast = forecast + self.dropout(update)
            x = torch.cat((x[:, :, :-n_forecast], forecast), dim=2)
        return (x + self.dropout(self.ffn(self.ffn_norm(x)))).masked_fill(~mask, 0)


class ForecastTokenEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        p = config.forecast_patch_len
        if not isinstance(p, int) or p < 1 or config.pred_len < 1 or config.pred_len % p:
            raise ValueError('forecast_patch_len must be positive and divide pred_len exactly')
        if config.encoder_layers < 1:
            raise ValueError('encoder_layers must be at least 1')
        self.patch_len, self.n_forecast = p, config.pred_len // p
        self.n_vars = config.enc_in
        self.forecast_tokens = nn.Parameter(torch.randn(self.n_forecast, config.d_model)*0.02)
        self.channel_identity = nn.Parameter(torch.randn(config.enc_in, config.d_model)*0.02)
        self.blocks = nn.ModuleList([ForecastTokenBlock(config) for _ in range(config.encoder_layers)])
        self.output_norm = nn.LayerNorm(config.d_model)
        self.head = nn.Sequential(nn.Dropout(config.head_dropout), nn.Linear(config.d_model, p))

    def forward(self, tokens, valid, centers, input_length):
        b, c, k, d = tokens.shape
        if c != self.n_vars:
            raise ValueError('Input channel count must match enc_in and channel identity table')
        future = self.forecast_tokens[None, None].expand(b, c, -1, -1)
        x = torch.cat((tokens, future), 2)
        valid = torch.cat((valid, torch.ones(b, c, self.n_forecast,
                          dtype=torch.bool, device=tokens.device)), 2)
        # Zero-based sample centers: a patch of points 0..p-1 has center (p-1)/2.
        future_centers = input_length + torch.arange(self.n_forecast,
            device=tokens.device, dtype=torch.float32)*self.patch_len + (self.patch_len-1)/2
        positions = torch.cat((centers, future_centers[None, None].expand(b, c, -1)), 2)
        for block in self.blocks:
            x = block(x, valid, positions, self.n_forecast, self.channel_identity)
        output = self.head(self.output_norm(x[:, :, -self.n_forecast:]))
        return output.reshape(b, c, -1).transpose(1, 2)
