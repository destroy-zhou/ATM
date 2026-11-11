import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from einops import rearrange

#MLP block with silu activation
class MLP_block(nn.Module):
    def __init__(self, d_model, dropout=0.0):
        super(MLP_block, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x):
        x = self.mlp(x)

        return x

class diffusion_timestamp_embeder(nn.Module):
    def __init__(self, d_model, max_steps = 1000, dropout=0.0):
        super(diffusion_timestamp_embeder, self).__init__()
        self.d_model = d_model
        self.max_steps = max_steps

        pe = torch.zeros(max_steps, d_model)
        position = torch.arange(0, max_steps, dtype=torch.float)[:, None]

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(max_steps) / d_model))[None, :]
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe) # [max_steps, d_model]

        self.mlp = MLP_block(d_model, dropout)

    def forward(self, denoise_t):
        #denoise_t: [*] -> [*, d_model], int tensor
        position_embed = self.pe[denoise_t, :]
        position_embed = self.mlp(position_embed)

        return position_embed

class DMLP_block(nn.Module):
    """
    A Diffusion MLP block
    """
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.mlp = MLP_block(d_model, dropout)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 3 * d_model, bias=True)
        )

    def forward(self, x, c):
        #x: [batch_size, patch_num, d_model], x has already been projected from patch_size to d_model
        #c: [batch_size, patch_num, d_model]
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(3, dim=-1)

        skip_x = x
        x = self.norm(x) * (1 + scale_mlp) + shift_mlp
        x = skip_x + gate_mlp * self.mlp(x)

        return x

class FinalLayer(nn.Module):
    """
    The final layer
    """
    def __init__(self, d_model, patch_size, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2 * d_model, bias=True)
        )
        self.out_project = nn.Linear(d_model, patch_size, bias=True)

    def forward(self, x, c):
        #x: [batch_size, patch_num, d_model]
        #c: [batch_size, patch_num, d_model]

        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)

        x = self.norm(x) * (1 + scale) + shift
        x = self.out_project(x) # [batch_size, patch_num, patch_size]

        return x

class Patch_Consistent_MLP(nn.Module):
    def __init__(self,
        max_diffusion_steps,
        d_condition,
        d_diffusion,
        patch_size,
        num_layers,
        radius,
        dropout=0.0,
        *args, **kwargs
    ):
        super(Patch_Consistent_MLP, self).__init__()

        self.max_diffusion_steps = max_diffusion_steps
        self.patch_size = patch_size
        self.d_condition = d_condition
        self.d_diffusion = d_diffusion
        self.num_layers = num_layers
        self.radius = radius
        self.dropout = dropout

        #to embed the patch itself
        self.patch_embedder = nn.Linear(patch_size, d_diffusion, bias=False)

        #to embed the adjacent patches (previous and next, not including the current one)
        self.prev_patches_embedder = nn.Conv1d(patch_size, d_diffusion, kernel_size=self.radius, padding=self.radius, bias=False)
        self.next_patches_embedder = nn.Conv1d(patch_size, d_diffusion, kernel_size=self.radius, padding=self.radius, bias=False)

        self.t_embedder = diffusion_timestamp_embeder(d_diffusion, max_diffusion_steps, dropout)
        self.condition_embedder = nn.Linear(d_condition, d_diffusion)

        self.blocks = nn.ModuleList([DMLP_block(d_diffusion, dropout) for _ in range(num_layers)])

        self.final_layer = FinalLayer(d_diffusion, patch_size, dropout)


    def forward(self, noisy_patch, condition, diffusion_t):
        """
        Forward pass of patch denoiser
        noisy_patch: [batch_size, patch_num, patch_size]
        condition: [batch_size, patch_num, d_condition], out put of Transformer decoder
        diffusion_t: [batch_size], int tensor
        """
        batch_size, patch_num, patch_size = noisy_patch.shape

        t = self.t_embedder(diffusion_t) # [batch_size, d_diffusion]
        expanded_t = t[:, None, :].expand(-1, patch_num, -1) # [batch_size, patch_num, d_diffusion]
        dec_condition = self.condition_embedder(condition) # [batch_size, patch_num, d_diffusion]

        previous_patch_emb = self.prev_patches_embedder(noisy_patch.transpose(1, 2)).transpose(1, 2)[:, :-self.radius - 1, :]
        next_patch_emb = self.next_patches_embedder(noisy_patch.transpose(1, 2)).transpose(1, 2)[:, self.radius + 1:, :]
        adj_patch_emb = previous_patch_emb + next_patch_emb

        x = self.patch_embedder(noisy_patch)
        condition = dec_condition + expanded_t + adj_patch_emb

        for block in self.blocks:
            x = block(x, condition)

        x = self.final_layer(x, condition)

        return x
