from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer
from layers.Embed import PatchEmbedding, PositionalEmbedding, TimeFeatureEmbedding
import transformers
from layers.StandardNorm import Normalize
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Time_Tokenizer import TimeTokenizer, FFNLayer
from utils.tools import init_LLM

transformers.logging.set_verbosity_error()

class TemporalBlock(nn.Module):
    def __init__(self, d_model, inter_dim):
        super().__init__()
        self.d_model = d_model
        self.inter_dim = inter_dim
        self.gate_proj = nn.Linear(d_model, self.inter_dim, bias=False)
        self.up_proj = nn.Linear(d_model, self.inter_dim, bias=False)
        self.down_proj = nn.Linear(self.inter_dim, d_model, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))

class ExpertsLayer(nn.Module):
    def __init__(self, d_model, num_experts, top_k, inter_dim):
        super().__init__()
        self.top_k = top_k
        self.d_model = d_model
        self.num_experts = num_experts

        inter_dim = inter_dim // top_k

        # gating
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [TemporalBlock(
                d_model=d_model,
                inter_dim=inter_dim,               
            ) for _ in range(self.num_experts)]
        )

        self.shared_expert = TemporalBlock(
            d_model=d_model,
            inter_dim=inter_dim,
        )
        self.shared_expert_gate = torch.nn.Linear(d_model, 1, bias=False)

    def forward(self, hidden_states):
        """ """
        batch_size, input_len, hidden_dim = hidden_states.shape
        # hidden_states -> (batch * input_len, hidden_dim)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # gate_vals -> (batch * input_len, n_experts)
        gate_vals = self.gate(hidden_states)

        gate_weights = F.softmax(gate_vals, dim=1, dtype=torch.float)
        #print("\ngate_weights:",gate_weights.shape,"\n")
        # selected_experts.shape: (batch * input_len, top_k)
        gate_weights, selected_experts = torch.topk(gate_weights, self.top_k, dim=-1)
        gate_weights = gate_weights.to(hidden_states.dtype) 

        final_hidden_states = torch.zeros(
            (batch_size * input_len, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # 计算每个expert被哪些时刻激活
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        #print(expert_mask,"\n")

        # 枚举expert，计算
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            # expert排名，时刻
            idx, top_x = torch.where(expert_mask[expert_idx])
            #print("idx:", idx, "\ntop_x:", top_x,"\n")

            # 每个时刻，对应专家与权重相乘
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * gate_weights[top_x, idx, None]

            # 进行累加
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        shared_expert_output = self.shared_expert(hidden_states)
        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states)) * shared_expert_output

        final_hidden_states = final_hidden_states + shared_expert_output

        final_hidden_states = final_hidden_states.reshape(batch_size, input_len, hidden_dim)
        return final_hidden_states, gate_vals

# 计算expert辅助损失
def load_balancing_loss_func(gate_vals, top_k, num_experts):
    concatenated_gate_vals = torch.cat([layer_gate.to(layer_gate.device) for layer_gate in gate_vals], dim=0)
    #print("\nconcatenated_gate_vals:",concatenated_gate_vals.shape,"\n")
    gate_weights = torch.nn.functional.softmax(concatenated_gate_vals, dim=-1)

    _, selected_experts = torch.topk(gate_weights, top_k, dim=-1)
    #print("\nselected_experts:",selected_experts.shape,"\n")

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    # 计算每个expert的token占比
    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

    # 计算每个expert的平均路由概率
    router_prob_per_expert = torch.mean(gate_weights, dim=0)
    # print(tokens_per_expert.shape, router_prob_per_expert.shape)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(dim=0))

    return overall_loss * num_experts

class FFNLayer(nn.Module):
    def __init__(self, d_model, inter_dim, dropout=0.1, activation="gelu"):
        super().__init__()
        self.w1 = nn.Linear(d_model, inter_dim)
        self.w2 = nn.Linear(inter_dim, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.relu if activation == "relu" else F.gelu

    def forward(self, x):
        x = self.w2(self.dropout(self.act(self.w1(x))))
        return x, None

class decoder(nn.Module):
    def __init__(self, c_attention, attention, p_attention, experts_layer, d_model, inter_dim, dropout, activation="gelu"):
        super().__init__()
        self.c_attention = c_attention
        self.attention = attention
        self.p_attention = p_attention
        self.experts_layer = experts_layer
        self.ffn_layer = FFNLayer(d_model, inter_dim, dropout, 'gelu')
        self.d_model = d_model
        self.norm0 = nn.LayerNorm(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        #self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, pos_mask, use_moe=True):
        B, C, L, D = x.shape
        x = x.permute(0,2,1,3).contiguous().view(-1, C, D)
        # print("x:", x.shape)

        delta_x, atten = self.c_attention(x, x, x, attn_mask=None)
        x = x + self.dropout(delta_x)
        x = self.norm0(x)
        x = x.view(B, L, C, D).permute(0,2,1,3).contiguous()
        # print("x:", x.shape)

        x = x.view(-1, L, D)
        # print(x.shape)
        atten_mask = None
        if pos_mask != None:
            atten_mask = ~pos_mask #.unsqueeze(1).expand(-1, pos_mask.shape[1], -1)
        # print("atten_mask:", atten_mask.shape)
        delta_x, atten = self.p_attention(x, x, x, attn_mask=atten_mask)
        x = x + self.dropout(delta_x)
        x = self.norm2(x)
        # print("x:", x.shape)

        delta_x, atten = self.attention(x, x, x, attn_mask=None)
        x = x + self.dropout(delta_x)
        x = self.norm1(x)
        # print('decoder:', x.shape)

        if use_moe:
            y, gate_vals = self.experts_layer(x)
        else:
            y, gate_vals = self.ffn_layer(x)

        x = self.norm3(x + y)
        x = x.view(B, C, L, D)

        return x, atten, gate_vals


class FlattenHead(nn.Module):
    def __init__(self, seq_len, n_vars, nf, d_model, inter_dim, target_window, head_dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.nf = nf
        self.flatten = nn.Flatten(start_dim=-2)
        ks = [[4,4]]
        layers = []
        for kernel_size, stride in ks:
            layers.append(nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, stride=stride, groups=d_model))
            #layers.append(nn.GELU())
        self.conv = nn.Sequential(*layers)

        input_len = seq_len
        for kernel_size, stride in ks:
            input_len = (input_len - kernel_size) // stride + 1    

        nf = seq_len * d_model
        self.linear = nn.Linear(nf, target_window)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(target_window)
        self.dropout = nn.Dropout(head_dropout)
        print("head_dropout:", head_dropout)
        #self.dropout0 = nn.Dropout(head_dropout)

    def forward(self, x):
        b, c, l, d = x.shape
        #print("out:",x.shape)
        x = x.view(b*c, l, -1).permute(0,2,1)
        # print("out:",x.shape)
        x = self.flatten(x)
        #x = self.dropout(x)
        # print("out:",x.shape)
        x = self.linear(x)
        x = self.dropout(x)
        #print("out:",x.shape)
        x = x.view(b, c, -1)
        #x = self.norm(x)
        return x


class Model(nn.Module):
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model
        self.patch_size = 16
        self.stride = 8
        self.mixed_precision = configs.mixed_precision
        self.threshold_ratio = configs.threshold_ratio
        self.model_verbose = configs.model_verbose
        
        # 动态Patch
        self.time_tokenizer = TimeTokenizer(configs.seq_len, configs.enc_in, configs.d_model, configs.inter_dim, configs.n_querys, 
                                        configs.threshold_ratio, configs.n_heads, configs.dropout, configs.aux_loss, configs.conv_layers,
                                        configs.high_freq, configs.prob_bias, configs.prob_bias_end, configs.use_time_tokenizer, configs.model_verbose)
        self.position_embedding = PositionalEmbedding(d_model=self.d_model)
        patch_mask = True if configs.use_time_tokenizer != 0 else False
        # print(patch_mask)
        self.decoder = nn.ModuleList(
            [decoder(
                AttentionLayer(FullAttention(mask_flag=False, attention_dropout=configs.dropout, output_attention=True), configs.d_model, configs.n_heads),
                AttentionLayer(FullAttention(mask_flag=False, attention_dropout=configs.dropout, output_attention=True), configs.d_model, configs.n_heads),
                AttentionLayer(FullAttention(mask_flag=patch_mask, attention_dropout=0., output_attention=True), configs.d_model, configs.n_heads),
                ExpertsLayer(configs.d_model, configs.num_experts, configs.top_k, configs.inter_dim),
                configs.d_model,
                configs.inter_dim,
                configs.dropout,
                configs.activation
                ) for l in range(configs.d_layers)]
            )
        self.norm = nn.LayerNorm(self.d_model)
      
        self.patch_nums = int((configs.seq_len - self.patch_size) / self.stride + 2)
        self.head_nf = self.d_model * configs.seq_len

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.seq_len, configs.enc_in, self.head_nf, configs.d_model, configs.inter_dim, self.pred_len,
                                                 head_dropout=configs.head_dropout)
        else:
            raise NotImplementedError

        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        # self.temporal_embedding = TimeFeatureEmbedding(d_model=configs.d_model, embed_type=configs.embed, freq=configs.freq)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, train=True):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out, patch_loss = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, train)
            return dec_out[:, -self.pred_len:, :], patch_loss
        return None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, train=True):

        x_enc = self.normalize_layers(x_enc, 'norm')
        #print("x_enc:", x_enc.shape)

        B, T, C = x_enc.size()
        n_vars = C
        # x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * C, T, 1).contiguous()

        # 进行分块 (batch * variate, seq_len, hidden_dim)
        x, aux_loss, patch_pos_mask = self.time_tokenizer(x_enc, train, self.configs.use_semantic_pe)
        # print('patch:', x.shape)
        # x = x + self.temporal_embedding(x_mark_enc)
        # print(self.temporal_embedding(x_mark_enc).shape)
        x = x + self.position_embedding(x)
        # print('patch:', x.shape)
        x = x.view(B, C, T, -1)
        # print('patch:', x.shape)
        
        all_gate_vals = []
        for decoder_layer in self.decoder:
            x, atten, gate_vals = decoder_layer(x, patch_pos_mask, self.configs.use_moe)
            all_gate_vals += [gate_vals,]        
        x = self.norm(x)        
        # print('x:', x.shape)
        x = x.reshape(B, C, T, -1)
        # print('x:', x.shape) 

        # 计算路由辅助损失
        if train and self.configs.use_moe and self.configs.apply_router_aux_loss:
            if self.model_verbose:
                print("aux_loss:", aux_loss.item(), end=', ')
            aux_loss *= self.configs.aux_loss_factor
            router_aux_loss = load_balancing_loss_func(all_gate_vals, self.configs.top_k,
                self.configs.num_experts)
            if torch.isnan(router_aux_loss):
                router_aux_loss = 0
            aux_loss = aux_loss + self.configs.router_aux_loss_factor * router_aux_loss
            if self.model_verbose:
                print("aux_loss:", aux_loss.item())
        
        dec_out = self.output_projection(x) 
        dec_out = dec_out.permute(0, 2, 1).contiguous()
        # print("dec_out:", dec_out.shape)

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out, aux_loss