"""EMAformer network adapted from the official AAAI 2026 implementation.

Source: https://github.com/PlanckChang/EMAformer
"""

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1, output_attention=False):
        super().__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask=None):
        _, _, _, head_dim = queries.shape
        scale = self.scale or 1.0 / sqrt(head_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attention = self.dropout(torch.softmax(scale * scores, dim=-1))
        output = torch.einsum("bhls,bshd->blhd", attention, values)
        return output.contiguous(), attention if self.output_attention else None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = d_model // n_heads
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, head_dim * n_heads)
        self.key_projection = nn.Linear(d_model, head_dim * n_heads)
        self.value_projection = nn.Linear(d_model, head_dim * n_heads)
        self.out_projection = nn.Linear(head_dim * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask=None):
        batch, query_len, _ = queries.shape
        key_len = keys.shape[1]
        queries = self.query_projection(queries).view(batch, query_len, self.n_heads, -1)
        keys = self.key_projection(keys).view(batch, key_len, self.n_heads, -1)
        values = self.value_projection(values).view(batch, key_len, self.n_heads, -1)
        output, attention = self.inner_attention(queries, keys, values, attn_mask)
        return self.out_projection(output.reshape(batch, query_len, -1)), attention


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        attention_output, attention = self.attention(x, x, x)
        x = x + self.dropout(attention_output)
        residual = x = self.norm1(x)
        residual = self.dropout(self.activation(self.conv1(residual.transpose(1, 2))))
        residual = self.dropout(self.conv2(residual).transpose(1, 2))
        return self.norm2(x + residual), attention


class Encoder(nn.Module):
    def __init__(self, layers, d_model):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        attentions = []
        for layer in self.layers:
            x, attention = layer(x)
            attentions.append(attention)
        return self.norm(x), attentions


class EMAformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pred_len = config.pred_len
        self.output_attention = config.output_attention
        self.use_norm = bool(config.use_norm)
        self.d_model = config.d_model
        self.cycle = config.cycle
        self.enc_in = config.enc_in

        self.value_embedding = nn.Linear(config.seq_len, config.d_model)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            attention_dropout=config.dropout,
                            output_attention=config.output_attention,
                        ),
                        config.d_model,
                        config.n_heads,
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation,
                )
                for _ in range(config.e_layers)
            ],
            config.d_model,
        )
        self.projector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.output_proj_dropout),
            nn.Linear(config.d_model * 2, config.d_model * 4),
            nn.GELU(),
            nn.Dropout(config.output_proj_dropout),
            nn.Linear(config.d_model * 4, config.pred_len),
        )

        self.channel_embedding = nn.Parameter(torch.empty(config.enc_in, config.d_model))
        self.phase_embedding = nn.Embedding(config.cycle, config.d_model)
        self.joint_embedding = nn.Embedding(config.cycle, config.enc_in * config.d_model)
        nn.init.xavier_normal_(self.channel_embedding)
        nn.init.xavier_normal_(self.phase_embedding.weight)
        nn.init.xavier_normal_(self.joint_embedding.weight)

    def forecast(self, x, phase):
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        batch, _, channels = x.shape
        if channels != self.enc_in:
            raise ValueError(f"EMAformer was built for {self.enc_in} channels, got {channels}")

        embedded = self.embedding_dropout(self.value_embedding(x.permute(0, 2, 1)))
        channel_embedding = self.channel_embedding.unsqueeze(0).expand(batch, -1, -1)
        phase_embedding = self.phase_embedding(phase).unsqueeze(1).expand(-1, channels, -1)
        joint_embedding = self.joint_embedding(phase).reshape(batch, channels, self.d_model)
        embedded = embedded + channel_embedding + phase_embedding + joint_embedding
        encoder_input = embedded

        embedded, attentions = self.encoder(embedded)
        output = self.projector(embedded + encoder_input).permute(0, 2, 1)

        if self.use_norm:
            output = output * stdev[:, 0, :].unsqueeze(1)
            output = output + means[:, 0, :].unsqueeze(1)
        return output, attentions

    def forward(self, x, phase):
        output, attentions = self.forecast(x, phase)
        if self.output_attention:
            return output[:, -self.pred_len :, :], attentions
        return output[:, -self.pred_len :, :]
