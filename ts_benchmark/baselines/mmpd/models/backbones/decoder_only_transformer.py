from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops import rearrange
import torch
from math import ceil

class learnable_position_embedding(nn.Module):
    def __init__(self, d_model, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.position_embedding = nn.Parameter(torch.randn(max_len, d_model), requires_grad=True) #add one more for zero-padding

    def forward(self, seq_idxs):
        """Positional encoding

        Args:
            seq_idxs shape: [batch_size, patch_num]

        Returns:
            torch.tensor: output position embedding with shape [batch_size, patch_num, d_model]
        """
        batch_size = seq_idxs.shape[0]

        position_embedding_expand = self.position_embedding[None, :, :].expand(batch_size, -1, -1)
        idxs_expand = seq_idxs[:, :, None].expand(-1, -1, self.d_model)
        pe = position_embedding_expand.gather(1, idxs_expand)

        return pe


class Transformer(nn.Module):
    def __init__(self, n_layers=3, d_model=256, n_heads=4,  d_ff=512, dropout=0.):
        super(Transformer, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = dropout

        encoder_layer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first = True)
        self.encoder_layers = TransformerEncoder(encoder_layer, n_layers)

    def forward(self, seqs):

        encoded_seqs = self.encoder_layers(seqs)

        return encoded_seqs

class DecoderOnlyTransformer(nn.Module):
    def __init__(self, configs):
        super(DecoderOnlyTransformer, self).__init__()

        self.in_len = configs.in_len
        self.out_len = configs.out_len
        self.patch_size = configs.patch_size
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.n_heads = configs.n_heads
        self.d_layers = configs.d_layers
        self.dropout = configs.dropout

        self.patch_embedding = nn.Linear(self.patch_size, self.d_model, bias=False)

        self.position_embedding = learnable_position_embedding(self.d_model)
        self.learnable_patch = nn.Parameter(torch.randn(self.d_model))


        self.decoder = Transformer(self.d_layers, self.d_model, self.n_heads, self.d_ff, self.dropout)

    def forward(self, x_seq, *args, **kwargs):
        """Forward pass

        Args:
            x_seq scaled ts, shape: [batch_size, data_dim, in_len]

        Returns:
            torch.tensor: tokens of [batch_size, out_patch_num, d_model]
        """

        batch_size, data_dim, seq_len = x_seq.shape
        x_seq = rearrange(x_seq, 'b d l -> (b d) l')

        #pad if not multiple of patch_size
        point_to_pad = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if point_to_pad > 0:
            x_seq = torch.cat([x_seq[:, 0:1].expand(-1, point_to_pad), x_seq], dim=1)
            seq_len = x_seq.shape[1]

        input_patch_num = seq_len // self.patch_size
        output_patch_num = ceil(self.out_len / self.patch_size)
        flatten_batch_size = x_seq.shape[0]

        #patchify
        patch_seq = rearrange(x_seq, 'b (n p) -> b n p', p = self.patch_size)

        #patch embedding
        patch_embed = self.patch_embedding(patch_seq) # [batch_size, input_patch_num, d_model]
        in_idxs = torch.arange(input_patch_num)[None, :].expand(flatten_batch_size, -1).to(x_seq.device)
        in_pos_embed = self.position_embedding(in_idxs)
        input_embed = patch_embed + in_pos_embed

        #decoder
        out_idxs = torch.arange(input_patch_num, input_patch_num + output_patch_num)[None, :].expand(flatten_batch_size, -1).to(x_seq.device)
        out_pos_embed = self.position_embedding(out_idxs)
        out_patch_embed = out_pos_embed + self.learnable_patch[None, None, :]
        dec_in = torch.cat([input_embed, out_patch_embed], dim = 1)
        dec_out = self.decoder(dec_in)[:, -output_patch_num:, :]

        dec_out = rearrange(dec_out, '(batch_size data_dim) out_len d_model -> batch_size data_dim out_len d_model', data_dim = data_dim)
        return dec_out  # [batch_size, data_dim, out_patch_num, d_model]
