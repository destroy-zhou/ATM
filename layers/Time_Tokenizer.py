import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.SelfAttention_Family import AttentionLayer, FullAttention, SemanticExtractor
from layers.Embed import PositionalEmbedding
from layers.Aux_Loss import calcute_aux_loss

iter = 500

class TimeTokenizer(nn.Module):
	def __init__(self, seq_len, c_in, d_model, inter_dim, n_querys, threshold_ratio, n_heads=8, dropout=0.1, aux_loss=1, conv_layers=4, high_freq=200, prob_bias=0.01, prob_bias_end=150, use_time_tokenizer=1, model_verbose=0):
		super().__init__()
		self.seq_len = seq_len
		self.d_model = d_model
		self.threshold_ratio = threshold_ratio
		self.c_embed = nn.Linear(1, d_model)
		# self.c_embed = TemporalInputEmbedding(1, d_model)
		self.iter = iter
		self.high_freq = high_freq
		self.prob_bias = prob_bias
		self.prob_bias_iter = 0
		self.prob_bias_end = prob_bias_end
		# self.dropout = dropout
		self.aux_loss = aux_loss
		self.position_embedding = PositionalEmbedding(d_model=d_model)
		self.use_time_tokenizer = use_time_tokenizer
		self.model_verbose = model_verbose

		ks = [[2 **(i+1),2 **(i+1)] for i in range(conv_layers)]
		print(ks)
		input_lens = []
		input_len = seq_len
		for kernel_size, stride in ks:
			input_lens.append(input_len)
			input_len = (seq_len - kernel_size) // stride + 1	
			# input_len = int(np.ceil((seq_len - kernel_size) / stride) + 1)

		self.convs = nn.ModuleList([
			ConvBlock(seq_len, input_len, d_model, inter_dim, kernel_size, stride, n_heads, dropout)
			for (kernel_size, stride), input_len in zip(ks, input_lens)
			])
		self.attens = nn.ModuleList()
		for i, ((kernel_size, stride), input_len) in enumerate(zip(ks, input_lens)):
			first = True if i==len(ks)-1 else False
			atten_block = AttenBlock(seq_len, input_len, d_model, inter_dim, kernel_size, stride, n_heads, dropout, first=first)
			self.attens.append(atten_block)
			
		self.attn_block = AttenBlock(seq_len, seq_len, d_model, inter_dim, 1, 1, n_heads, dropout, last=True)
		self.prob_proj = nn.Linear(d_model, 1, bias=False)  #, bias=False
		nn.init.normal_(self.prob_proj.weight, mean=-0.1, std=0.01)
		self.patch_d_model = d_model
		self.patch_proj = FFNLayer(1, seq_len, inter_dim, self.patch_d_model, dropout=dropout)
		#self.patch_proj = nn.Linear(seq_len, self.patch_d_model)
		self.dropout = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.embed = nn.Linear(1, d_model)
		# self.embed = TemporalInputEmbedding(1, d_model)
		self.SemanticExtractor = SemanticExtractor(self.patch_d_model, n_querys, 0., dropout, mask_flag=True)
		self.pe_embed = nn.Linear(d_model, d_model)
		# self.pe_embed = TemporalBlock(d_model, inter_dim, d_model, 0.)
		if use_time_tokenizer == 2:
			self.mlp_tokenizer = FFNLayer(1, seq_len, seq_len, seq_len, dropout, norm='l')

	def forward(self, x, train=True, use_semantic_pe=1):
		b, l, c = x.shape
		
		# querys = self.decoder(x)
		# querys = querys.unsqueeze(0)

		x = x.permute(0, 2, 1).contiguous().view(-1, l).contiguous()
		#print("x:", x.shape)
		x_embed = self.c_embed(x.unsqueeze(-1))
		x_embed = self.dropout(x_embed)
		patches = [x_embed]
		patch = x_embed

		# use time_tokenizer
		if self.use_time_tokenizer == 1:
			for i, conv in enumerate(self.convs):
				patch = conv(x_embed)
				#print("patch:",patch.shape)
				patches.append(patch)

			attens = None
			last_p = None
			for i, atten in enumerate(self.attens[::-1]):
				last_p = atten(patches[len(patches)-i-1], last_p)
			patch = self.attn_block(patches[0], last_p)
		elif self.use_time_tokenizer == 2:  # use MLP
			probs = self.mlp_tokenizer(patch.permute(0,2,1)).permute(0,2,1)
		else:
			return patch, torch.tensor(0.), None

		# Make sure that the probs < 0.5 at the beginning
		if train:
			if self.prob_bias_iter == self.prob_bias_end:
				probs = self.prob_proj(patch)
			else:
				probs = self.prob_proj(patch + self.prob_bias)
				self.prob_bias_iter += 1
		else:
			probs = self.prob_proj(patch)
		probs = F.sigmoid(probs)
		#print(probs)
		probs_hard = -probs.detach() + probs
		#probs_hard = probs
		probs_hard = probs_hard.squeeze(-1)
		#print("probs_hard:", probs_hard)
		#print(probs_hard.shape)
		x = x + probs_hard
		#print("x:",x.shape)
		#print(x)

		mask = probs[:,:-1] >= 0.5  # shape: [batch_size, seq_len-1]
		patch_counts = mask.sum(dim=1) + 1  # shape: [batch_size]
		max_patch_num = patch_counts.max().item()
		min_patch_num = patch_counts.min().item()
		if not train:
			self.iter -= 1
		if train or self.iter == 0:
			if self.model_verbose:
				print(min_patch_num, max_patch_num)

		x_embed = self.embed(x.unsqueeze(-1))
		x_embed = x_embed.view(-1, x_embed.shape[-1])
		# print('x_embed:', x_embed.shape)
		probs = probs.view(-1)
		# print('probs:', probs.shape)
		x = x.view(-1)
		# print("x:", x.shape)
		#print("x_embed:", x_embed.shape)
		#x_embed = self.dropout2(x_embed)
		if train and self.aux_loss:
			aux_loss = torch.tensor(1.).to(x.device)
		else:
			aux_loss = torch.tensor(0.000).to(x.device)
		x_out = torch.zeros(x_embed.shape).to(x.device)
		patches = []
		# mask = torch.zeros((b, max_patch_num), dtype=torch.bool).to(x.device) 
		# end_mask = torch.zeros((b, max_patch_num), dtype=torch.bool).to(x.device)

		start = 0; loss = 0
		condition = (probs[:-1] >= 0.5)
		end_condition = torch.zeros_like(condition, dtype=torch.bool)
		# 设置每第96个索引（即索引为95, 191, ...）为True
		end_condition[self.seq_len-1::self.seq_len] = True
		idx = torch.where(condition | end_condition)[0] + 1
		starts = torch.cat((torch.tensor([0]).to(x.device), idx)).unsqueeze(1)
		ends = torch.cat((idx, torch.tensor([len(probs)]).to(x.device))).unsqueeze(1)
		pos = torch.cat([starts, ends], dim=1)
		# print(pos[-10:])

		# 获取最大区间长度
		lengths = pos[:, 1] - pos[:, 0]
		max_len = lengths.max().item() #self.seq_len lengths.max().item()
		# 构建一个范围掩码 (1, max_len)
		range_matrix = torch.arange(max_len).unsqueeze(0).to(x.device)
		# mask: 哪些位置在实际区间内(N, max_len)
		pos_mask = range_matrix < lengths.unsqueeze(1)
		# print(pos_mask.shape)
		# 构建所有偏移索引 (N, max_len)
		indices = pos[:, 0].unsqueeze(1) + range_matrix
		# print(indices)
		indices += 1
		# print(indices)
		# 将 mask 无效的位置设为 0
		indices[~pos_mask] = 0
		# print(indices[:20])
		# 提取所有值 (N, max_len)
		# res = x[indices]
		# res[~pos_mask] = 0
		# print(res.shape)
		#print(res)
		zero_row = torch.zeros(1, x_embed.shape[-1]).to(x.device)  # 生成形状为 (1, 10) 的全零 tensor
		x_embed_padded = torch.cat([zero_row, x_embed], dim=0)
		x_out = x_embed_padded[indices]
		# x_out[~pos_mask] = 0
		# print("x_out:", x_out.shape)
		with torch.no_grad():
			inter_pos = torch.cat((starts.squeeze(-1), torch.tensor([len(probs)]).to(x.device)))
			# print('pe_pos:', pe_pos)
			# print('pe_pos:', pe_pos.shape)
			# 获取值是96的倍数的下标
			inter_starts = torch.nonzero(inter_pos % self.seq_len == 0).squeeze()
			# print("inter_starts:", inter_starts.shape)
			# print("inter_starts:", inter_starts)
			# 获取相邻下标之间的间隔
			intervals = inter_starts[1:] - inter_starts[:-1]
			# print('intervals:', intervals)
			# print(intervals.sum())
			# 构建一个范围掩码 (1, max_len)
			# print(intervals.max())
			inter_range_matrix = torch.arange(self.seq_len).unsqueeze(0).to(x.device)
			# mask: 哪些位置在实际区间内(N, max_len)
			inter_pos_mask = inter_range_matrix < intervals.unsqueeze(1)
			# print(inter_pos_mask)
			# print(inter_pos_mask.shape)
			inter_indices = inter_starts[:-1].unsqueeze(1) + inter_range_matrix
			# 将 mask 无效的位置设为 0
			inter_indices[~inter_pos_mask] = 0
			# print(inter_indices.shape)
			# print(inter_indices)

			patch_starts = starts % self.seq_len
			patch_ends = (ends - 1) % self.seq_len
			patch_range_matrix = torch.arange(self.seq_len).unsqueeze(0).to(x.device)
			# (b * num_patches, seq_len)
			patch_pos_mask = (patch_range_matrix >= patch_starts) & (patch_range_matrix <= patch_ends)
			# print(patch_pos_mask.shape)
			# print(patch_pos_mask)
			# (b*c ,l, l)
			position_to_patch = patch_pos_mask[inter_indices]
			# print(position_to_patch.shape)
			# print(position_to_patch)
			position_to_patch = position_to_patch.permute(1,0,2).contiguous().view(l, -1)
			# print(position_to_patch.shape)
			# print(position_to_patch)
			position_to_patch = position_to_patch.float().argmax(dim=0)  # (b * c * l)
			# print(position_to_patch.shape)
			# print(position_to_patch)
			position_to_patch = position_to_patch.view(b*c, -1)
			# print(position_to_patch.shape)
			# print(position_to_patch)
			# print("inter_starts:", inter_starts.shape)
			position_to_patch += inter_starts[:-1].unsqueeze(-1)
			# print(position_to_patch.shape)
			# print(position_to_patch)
			assert (position_to_patch < patch_pos_mask.shape[0]).all(), "Patch索引越界了！"
			# position_to_patch = 
			# 构建 batch 维度索引
			# batch_idx = torch.arange(b, device=x.device).unsqueeze(1).expand(-1, self.seq_len)
			# print(batch_idx)
			patch_pos_mask = patch_pos_mask[position_to_patch]
		# print(patch_pos_mask.shape) 
		# print(patch_pos_mask) 
		#print(patch)
		# patch_mean = x_out.sum(dim=1)
		# # print("patch_mean:", patch_mean.shape)
		# #print("sum:", patch)
		# lengths = lengths.unsqueeze(-1)
		# # print("lengths:", lengths.shape)
		# patch_mean = patch_mean  / lengths
		# print("patch_mean:", patch_mean.shape)
		# patches = patches.masked_fill_(mask, 0.)
		# patches[mask==0] = 0
		# print(patches[0][-5:])
		# (B*N, max_len)
		# x_patch = x_embed.view(b*c, l, -1)
		# # print("x_patch:", x_patch.shape)
		# # print(self.position_embedding(x_patch).shape)
		# x_patch = x_patch + self.position_embedding(x_patch)
		# x_patch = x_patch.view(-1, x_patch.shape[-1])[indices]
		_mask = ~pos_mask
		# (B*N, 1, max_len)
		atten_mask = _mask.unsqueeze(1) #.expand(-1, _mask.shape[1], -1)
		# print("atten_mask:", atten_mask.shape)
		# print(_mask[0])
		# print(atten_mask[0])
		patches = self.SemanticExtractor(x_out, query=None, attn_mask=atten_mask, apply_ortho_loss=False) #patch_mean.unsqueeze(1)
		# patches, ortho_loss = self.PatchAggregator(x_out, query=None, attn_mask=atten_mask, apply_ortho_loss=True)
		# print("patches:", patches.shape)
		# patches += patch_mean
		# semantic_pe = self.pe_embed(patches)
		# x_out1 = x_out + semantic_pe.unsqueeze(1)
		# x_out1 = x_out1[pos_mask]
		semantic_pe = patches.unsqueeze(1).expand(-1, pos_mask.shape[-1], -1)
		# print("semantic_pe:", semantic_pe.shape)
		semantic_pe = semantic_pe[pos_mask]
		# print("semantic_pe:", semantic_pe.shape)
		semantic_pe = self.pe_embed(semantic_pe)
		# print("pe:", pe.shape)
		x_out = x_out[pos_mask]
		if use_semantic_pe:
			x_out = x_out + semantic_pe #.unsqueeze(1)
		x_out = x_out.view(b*c, l, -1)
		# print('x_out:', x_out.shape)
		#aux_loss = torch.tensor(0.000)
		if train and self.aux_loss:
			aux_loss = torch.log(aux_loss) 
			if self.model_verbose:
				print(aux_loss.item())
			# print(aux_loss.item(), ortho_loss.item())
			# patches = torch.cat(patches, dim=0)
			# print(patches.shape)
			aux_loss += calcute_aux_loss(patches, self.threshold_ratio, train, self.iter, self.high_freq, self.model_verbose)
			# aux_loss += ortho_loss * 0.5
		#print("aux_loss:", aux_loss)
		if self.iter == 0:
			self.iter = iter
		
		#print("x:",x.shape)
		return x_out, aux_loss, patch_pos_mask

class ConvBlock(nn.Module):
	def __init__(self, seq_len, input_len, d_model, inter_dim, kernel_size, stride, n_heads, dropout=0.1):
		super().__init__()
		self.d_model = d_model
		self.seq_len = seq_len
		self.kernel_size = kernel_size
		self.stride = stride
		patch_num = (input_len - kernel_size) // stride + 1
		self.padding = 0 #kernel_size // 2
		if seq_len % kernel_size != 0:
			self.padding = kernel_size - seq_len % kernel_size
		self.conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=kernel_size, stride=stride, padding=0, groups=1, bias=False)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		#x = self.norm(x)
		# 在序列前填充0样本
		# x = nn.functional.pad(x,(0,0,self.padding,0))
		# print("x_pad:", x.shape)
		# print(x)
		x = self.conv(x.permute(0,2,1))
		return x.permute(0,2,1)

class AttenBlock(nn.Module):
	def __init__(self, seq_len, input_len, d_model, inter_dim, kernel_size, stride, n_heads, dropout=0.1, atten_dropout=0.1, last=False, first=False):
		super().__init__()
		self.d_model = d_model
		self.seq_len = seq_len
		self.last = last
		self.first = first
		# patch_num = int(np.ceil((seq_len - kernel_size) / stride) + 1)
		patch_num = (seq_len - kernel_size) // stride + 1
		print(patch_num)
		#self.atten = GetAttention(d_model, n_heads, dropout, bias=True)
		self.cross_atten = AttentionLayer(FullAttention(mask_flag=False, attention_dropout=atten_dropout, output_attention=True), d_model, n_heads, bias=False)
		self.atten = AttentionLayer(FullAttention(mask_flag=False, attention_dropout=atten_dropout, output_attention=True), d_model, n_heads, bias=False)
		self.kernel = max(1, 5 - int(np.log2(kernel_size))) * 2 + 1
		print('kernel:', self.kernel)
		self.padding = self.kernel // 2
		self.atten_embed = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=self.kernel, stride=1, padding=self.padding, groups=1, bias=False)
		self.atten_proj = nn.Linear(d_model, self.kernel, bias=False)
		# self.feature_proj = FFNLayer(patch_num, d_model * 2, inter_dim, d_model, dropout, norm='l', bias=False)
		self.feature_proj = TemporalBlock(d_model * 2, inter_dim, d_model, dropout)
		self.token_proj = nn.Linear(patch_num, input_len, bias=False)
		self.prob_proj = TemporalBlock(d_model, inter_dim, d_model, dropout)
		self.attn_dropout = nn.Dropout(dropout)
		# self.dropout0 = nn.Dropout(dropout)
		self.dropout = nn.Dropout(dropout)
		#self.norm = nn.BatchNorm1d(patch_num)
		# self.norm0 = nn.LayerNorm(d_model)
		self.norm = nn.LayerNorm(d_model)
		#self.norm2 = nn.LayerNorm(d_model)
		#self.norm3 = nn.LayerNorm(d_model)

	def forward(self, x, last_x):
		b, l, _ = x.shape
		#dx, atten = self.atten(x, x, x, attn_mask=None)
		#x = self.norm(dx + x)
		x = self.norm(x)
		# print("x:", x.shape)
		if last_x != None:
			last_x = self.atten_embed(last_x.permute(0,2,1)).permute(0,2,1)
			# print('last_x:', last_x.shape)
			atten = self.atten_proj(last_x)
			atten = self.attn_dropout(torch.softmax(atten, dim=-1))
			
			# 用 gather 按照上面构造的索引提取 B 中的子向量
			coarse_embed = F.pad(last_x, pad=(0,0,self.padding,self.padding))
			# print('coarse_embed:', coarse_embed.shape)
			# print(coarse_embed[0,:10])
			# 提取滑动窗口 (b, l, k, d)
			coarse_embed = coarse_embed.unfold(dimension=1, size=self.kernel, step=1).permute(0,1,3,2)
			# print('coarse_embed:', coarse_embed.shape)
			# print(coarse_embed[0,:2])
			# 点积 + 求和 -> (b, l ,d)
			coarse_embed = torch.einsum('blk,blkd->bld', atten, coarse_embed)
			# print('coarse_embed:', coarse_embed.shape)

			fine_embed = F.pad(x, pad=(0,0,self.padding,self.padding))
			# print('fine_embed:', fine_embed.shape)
			# 提取滑动窗口 (b, l, k, d)
			fine_embed = fine_embed.unfold(dimension=1, size=self.kernel, step=1).permute(0,1,3,2)
			# print('fine_embed:', fine_embed.shape)
			# print(coarse_embed[0,:2])
			# 点积 + 求和 -> (b, l ,d)
			fine_embed = torch.einsum('blk,blkd->bld', atten, fine_embed)
			# print('fine_embed:', fine_embed.shape)
			x = torch.cat([coarse_embed, fine_embed], dim=-1)
			# print("x:", x.shape)
			out = self.feature_proj(x)
		else:
			out, atten = self.atten(x, x, x, attn_mask=None)
			# out = x + self.dropout0(out)
			# out = self.norm0(out)
			out = self.prob_proj(out)
		if not self.last:
			out = self.token_proj(out.permute(0,2,1)).permute(0,2,1)
			out = self.dropout(out)
		#out = self.token_proj(atten.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
		#print("out:", out.shape)
		return out

class TemporalBlock(nn.Module):
	def __init__(self, input_dim, inter_dim, d_model, dropout=0.1):
		super().__init__()
		self.d_model = d_model
		self.inter_dim = inter_dim
		self.gate_proj = nn.Linear(input_dim, self.inter_dim, bias=False)
		self.up_proj = nn.Linear(input_dim, self.inter_dim, bias=False)
		self.down_proj = nn.Linear(self.inter_dim, d_model, bias=False)
		self.act_fn = nn.SiLU()
		self.dropout = nn.Dropout(dropout)
		self.norm = nn.LayerNorm(input_dim)

	def forward(self, x):
		# x = self.norm(x)
		y = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
		# y = self.dropout(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
		# y = self.down_proj(y)
		x = self.dropout(y)
		# x = self.norm(x + y)
		return x
	
class TemporalInputEmbedding(nn.Module):
	def __init__(self, input_dim, d_model):
		super().__init__()
		self.d_model = d_model
		self.gate_proj = nn.Linear(input_dim, d_model, bias=False)
		self.up_proj = nn.Linear(input_dim, d_model, bias=False)
		self.act_fn = nn.SiLU()
		self.norm = nn.LayerNorm(input_dim)

	def forward(self, x):
		# x = self.norm(x)
		x = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
		return x

class FFNLayer(nn.Module):
	def __init__(self, input_len, input_dim, inter_dim, output_dim, dropout=0.1, activation="gelu", norm="l", bias=True):
		super().__init__()
		self.w1 = nn.Linear(input_dim, inter_dim, bias=bias)
		self.w2 = nn.Linear(inter_dim, output_dim, bias=bias)
		self.norm = None
		if norm == "l":
			self.norm = nn.LayerNorm(output_dim)
		elif norm == 'b':
			self.norm = nn.BatchNorm1d(input_len)
		self.dropout = nn.Dropout(dropout)
		self.act = F.relu if activation == "relu" else F.gelu

	def forward(self, x):
		x = self.w2(self.dropout(self.act(self.w1(x))))
		x = self.dropout(x)
		if self.norm != None:
			x = self.norm(x)
		#print(self.norm)
		return x