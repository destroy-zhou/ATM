import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from layers.Embed import PositionalEmbedding

class PatchAggregator0(nn.Module):
    def __init__(self, d_model, n_heads=8, attention_dropout=0.1, mask_flag=False):
        super().__init__()
        dim = d_model // n_heads
        
        self.query = nn.Parameter(torch.randn(1, 1, d_model))  # [1, 1, D]
        self.proj_q = nn.Linear(d_model, dim * n_heads)
        self.proj_k = nn.Linear(d_model, dim * n_heads)
        self.proj_v = nn.Linear(d_model, dim * n_heads)
        self.proj_0 = nn.Linear(dim * n_heads, d_model)
        self.dropout = nn.Dropout(attention_dropout)
        self.n_heads = n_heads
        self.mask_flag = mask_flag
    
    def forward(self, patch, query=None, attn_mask=None):  # [B, L, D]
        B, L, D = patch.shape
        H = self.n_heads
        if query == None:
            # Q = self.proj_q(self.query.expand(B, -1, -1))  # [B, 1, H, D]
            Q = self.query.expand(B, -1, -1).view(B, 1, H, -1)
        else:
            # Q = self.proj_q(query)
            Q = query
        K = self.proj_k(patch).view(B, L, H, -1)       # [B, L, H, D]
        V = self.proj_v(patch).view(B, L, H, -1)       # [B, L, H, D]
        # V = patch
        # print(Q.shape, K.shape)
        # [B, H, 1, D]
        attn_scores = torch.einsum("blhe,bshe->bhls", Q, K) / (K.size(-1) ** 0.5)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=patch.device).mask
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                #print("scores:",scores.shape)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, attn_scores.shape[1], -1, -1)
                #print("attn_mask:",attn_mask.shape)
                attn_scores.masked_fill_(attn_mask, -np.inf)

        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))
        # print("attn_weights:", attn_weights)
        patch_token = torch.einsum("bhls,bshd->blhd", attn_weights, V)
        patch_token = patch_token.view(B, 1, -1)
        patch_token = self.proj_0(patch_token)
        # print("patch_token:", patch_token.shape)
        return patch_token.squeeze(1)       # [B, D]

class SemanticExtractor(nn.Module):
    def __init__(self, d_model, n_querys, attention_dropout=0.1, dropout=0.1, mask_flag=False):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, n_querys, d_model))  # [1, Q, D]
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(attention_dropout)
        self.mask_flag = mask_flag
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.fusion_attn = nn.Linear(d_model, 1)
        self.attn_dropout = nn.Dropout(dropout)

    def orthogonality_loss(self, query):  # [B, Q, D]
        Q = F.normalize(query, dim=-1)  # normalize along D
        sim = torch.matmul(Q, Q.transpose(-1, -2))  # [B, Q, Q]
        I = torch.eye(sim.shape[-1], device=sim.device).unsqueeze(0)
        return ((sim - I) ** 2).mean()
    
    def forward(self, patch, query=None, attn_mask=None, apply_ortho_loss=False):  # [B, L, D]
        B, M, D = patch.shape
        patch = patch + self.position_embedding(patch) #.to(patch.device)
        if query == None:
            Q = self.query.expand(B, -1, -1)
        else:
            Q = query.repeat(B // query.shape[0], 1, 1)
        # print("Q:", Q.shape)
        Q_cat = Q.unsqueeze(-2)
        # print("Q:", Q.shape)
        patch = torch.cat([Q_cat, patch.unsqueeze(1).expand(-1, Q.shape[1], -1, -1)], dim=-2)   #[B, Q, M+1, D]
        # print(patch.shape)
        K = self.proj_k(patch)   # [B, L, D]
        V = self.proj_v(patch)                         # [B, L, D]
        # V = patch
   
        attn_scores = torch.einsum('bqd,bqkd->bqk', Q, K)  # [B, Q, M+1]
        attn_scores /= D ** 0.5
        # print("attn_scores:", attn_scores.shape)
        # print(attn_scores)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, M, device=patch.device).mask
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                #print("scores:",scores.shape)
                attn_mask = torch.cat([torch.zeros((B, 1, 1), dtype=torch.bool).to(K.device), attn_mask], dim=-1)
                # attn_mask = attn_mask.unsqueeze(1).expand(-1, attn_scores.shape[1], -1, -1)
                attn_mask = attn_mask.expand(-1, Q.shape[1], -1)
                # print("attn_mask:",attn_mask.shape)
                attn_scores.masked_fill_(attn_mask, -np.inf)

        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))           # [B, Q, L]
        # print("attn_weights:", attn_weights)

        patch_token = torch.einsum('bqk,bqkd->bqd', attn_weights, V)  # [B, Q, D]
        # patch_token = torch.matmul(attn_weights, V)  # [B, Q, D]
        # print(patch_token.shape)
        weights = self.attn_dropout(torch.softmax(self.fusion_attn(patch_token), dim=1))
        patch_token = torch.sum(patch_token * weights, dim=1).squeeze(-2)
        if apply_ortho_loss:
            ortho_loss = self.orthogonality_loss(patch_token) #Q
            return patch_token, ortho_loss   
        else:
            return patch_token

class PatchAggregator_f(nn.Module):
    def __init__(self, d_model, n_querys, attention_dropout=0.1, dropout=0.1, mask_flag=False):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, n_querys, d_model))  # [1, Q, D]
        self.proj_q = nn.Linear(d_model, d_model)
        self.proj_k = nn.Linear(d_model, d_model)
        self.proj_v = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(attention_dropout)
        self.mask_flag = mask_flag
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.fusion_attn = nn.Linear(d_model, 1)
        self.attn_dropout = nn.Dropout(dropout)

    def orthogonality_loss(self, query):  # [B, Q, D]
        Q = F.normalize(query, dim=-1)  # normalize along D
        sim = torch.matmul(Q, Q.transpose(-1, -2))  # [B, Q, Q]
        I = torch.eye(sim.shape[-1], device=sim.device).unsqueeze(0)
        return ((sim - I) ** 2).mean()
    
    def forward(self, patch, query=None, indices=None, pos_mask=None, apply_ortho_loss=False):  # [B, L, D]
        B, L, D = patch.shape
        q = self.query.shape[1]
        # patch = patch + self.position_embedding(patch) #.to(patch.device)
        if query == None:
            # Q = self.proj_q(self.query.expand(B, -1, -1))  # [B, Q, D]
            Q = self.query.expand(B, -1, -1)
        else:
            # Q = self.proj_q(query)
            Q = query.repeat(B // query.shape[0], 1, 1)
        # print("Q:", Q.shape)
        Q_cat = Q.unsqueeze(-2)
        # print("Q:", Q.shape)
        patch = torch.cat([Q_cat, patch.unsqueeze(1).expand(-1, Q.shape[1], -1, -1)], dim=-2)   #[B, Q, L+1, D]
        # print(patch.shape)
        K = self.proj_k(patch) #+ self.position_embedding(patch).to(patch.device)   # [B, L, D]
        V = self.proj_v(patch)                         # [B, L, D]
        # V = patch
   
        # attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (K.size(-1) ** 0.5) #.squeeze(-2)  # [B, Q, L]
        attn_scores = torch.einsum('bqd,bqkd->bqk', Q, K)  # [B, Q, M+1]
        attn_scores /= D ** 0.5
        # print("attn_scores:", attn_scores.shape)
        # print(attn_scores)
        q_token = attn_scores[0,:,0].unsqueeze(0).unsqueeze(-1)
        # print("q_token", q_token.shape)
        attn_scores = attn_scores[:,:,1:].permute(0,2,1).contiguous().view(-1, q)
        # print(attn_scores.shape)
        attn_scores = attn_scores[indices]   # [N, M, Q]
        attn_scores[~pos_mask] = -np.inf
        attn_scores = attn_scores.permute(0,2,1)
        # print(attn_scores.shape)
        attn_scores = torch.cat([q_token.expand(attn_scores.shape[0], -1, -1), attn_scores], dim=-1) # [N, Q, M+1]
        # print(attn_scores.shape)
        # print(patch_token)
        attn_weights = self.dropout(torch.softmax(attn_scores, dim=-1))           # [N, Q, M+1]
        # print(attn_weights)
        # q_token = attn_weights[0,:,0].unsqueeze(0).unsqueeze(-1)
        # print("q_token", q_token.shape)
        # attn_weights = attn_weights[:,:,1:].permute(0,2,1)
        # attn_weights = attn_weights[pos_mask]
        # print("attn_weights:", attn_weights.shape)
        # attn_weights = attn_weights.view(B, L, -1).permute(0,2,1)
        # print("attn_weights:", attn_weights.shape)
        # attn_weights = torch.cat([q_token.expand(attn_weights.shape[0], -1, -1), attn_weights], dim=-1).unsqueeze(-1)
        # print("attn_weights:", attn_weights.shape)
        # patch_token = attn_weights * V     # [B, Q, L+1, D]
        q_token = V[0,:,0].unsqueeze(0).unsqueeze(-2)
        # print("q_token", q_token.shape)
        V = V[:,:,1:].permute(0,2,1,3).contiguous().view(-1, q, D)
        # print(V.shape)
        V = V[indices]   # [N, M+1, Q, D]
        V[~pos_mask] = 0
        V = V.permute(0,2,1,3)
        # print(V.shape)
        V = torch.cat([q_token.expand(V.shape[0], -1, -1, -1), V], dim=-2) # [N, Q, M+1, D]
        # print("V:", V.shape)
        patch_token = torch.einsum('bqk,bqkd->bqd', attn_weights, V)
        # print(patch_token.shape)
        # q_token = patch_token[0,:,0].unsqueeze(0).unsqueeze(-2)
        # # print("q_token", q_token.shape)
        # patch_token = patch_token[:,:,1:].permute(0,2,1,3).contiguous().view(-1, q, D)
        # # print(patch_token.shape)
        # indices = indices #.unsqueeze(1).expand(-1, q, -1)
        # patch_token = patch_token[indices]   # [N, M, Q, D]
        # patch_token[~pos_mask] = 0
        # patch_token = patch_token.permute(0,2,1,3)
        # # print(patch_token.shape)
        # patch_token = torch.cat([q_token.expand(patch_token.shape[0], -1, -1, -1), patch_token], dim=-2)
        # # print(patch_token.shape)
        # # print(patch_token)
        # patch_token = torch.sum(patch_token, dim=-2)
        # print(patch_token.shape)
        weights = self.attn_dropout(torch.softmax(self.fusion_attn(patch_token), dim=1))
        patch_token = torch.sum(patch_token * weights, dim=1).squeeze(-2)
        if apply_ortho_loss:
            ortho_loss = self.orthogonality_loss(patch_token) #Q
            return patch_token, ortho_loss   
        else:
            return patch_token                 

class AttentionPooling(nn.Module):
    def __init__(self, d_model, attention_dropout=0.1, mask_flag=False):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, d_model))  # Learnable query
        self.dropout = nn.Dropout(attention_dropout)
        self.mask_flag = mask_flag

    def forward(self, patch, attn_mask):  # patch: (B, L, D)
        q = self.query.unsqueeze(0)  # (1, 1, D)
        k = v = patch   # (B, L, D)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (k.size(-1) ** 0.5)  # (B, 1, L)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device).mask
                scores.masked_fill_(attn_mask, -np.inf)
            else:
                #print("scores:",scores.shape)
                #attn_mask = attn_mask.unsqueeze(1).expand(-1, scores.shape[1], -1, -1)
                #print("attn_mask:",attn_mask.shape)
                scores.masked_fill_(attn_mask, -np.inf)

        atten = self.dropout(torch.softmax(scores, dim=-1))  # (B, 1, L)
        #print("atten:", atten)
        out = torch.matmul(atten, v).squeeze(1)  # (B, D)
        return out

class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device).mask

            scores.masked_fill_(attn_mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device).mask
                scores.masked_fill_(attn_mask, -np.inf)
            else:
                # print("scores:",scores.shape)
                attn_mask = attn_mask.unsqueeze(1).expand(-1, scores.shape[1], -1, -1)
                #print("attn_mask:",attn_mask.shape)
                scores.masked_fill_(attn_mask, -np.inf)
                #print("scores:", scores[0,:2])
                #print("attn_mask:", attn_mask[0,:2])
                # scores = scores * attn_mask
                #print("scores:", scores[0,:2])
                #sl

        #print(scores)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # print('A:', A)
        #print(A[0])
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, bias=True):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads, bias=bias)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads, bias=bias)
        self.value_projection = nn.Linear(d_model, d_values * n_heads, bias=bias)
        self.out_projection = nn.Linear(d_values * n_heads, d_model, bias=bias)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None
