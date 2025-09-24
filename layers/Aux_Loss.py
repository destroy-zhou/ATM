import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def calcute_aux_loss2(patches, threshold, train, iter):
	if train or iter == 0:
		print(patches.shape, end=' ')
	#print(patches[-5:])
	patches = patches[torch.any(patches != 0, dim=1)]
	#print(patches)
	word_num = patches.shape[0]
	if train or iter == 0:
		print(patches.shape)
	
	cosine_similaritys = F.cosine_similarity(patches.unsqueeze(1), patches.unsqueeze(0), dim=2).to(patches.device)
	patches = patches.unsqueeze(1)

	#mask = cosine_similaritys >= threshold
	#print("mask:", mask)
	#similar_nums = mask.sum(-1)
	#print("similar_nums:", similar_nums)
	cosine_similaritys[cosine_similaritys < threshold] = 0
	#print("cosine_similaritys:", cosine_similaritys)
	word_freq = cosine_similaritys.sum(-1)
	#print(word_freq.shape)
	#print(word_freq)
	#word_freq = word_freq / similar_nums
	#print(word_freq)
	word_freq = word_freq.mean()
	
	if train or iter == 0:
		print("word_freq:", word_freq.item(), "word_num:", word_num)
	aux_loss = torch.log(word_num / word_freq) 
	#print("word_freq:", word_freq.item(), "num_vectors:", num_vectors)
	return aux_loss

def calcute_aux_loss0(patches, threshold, train, iter):
	if train or iter == 0:
		print(patches.shape, end=' ')
	#print(patches[-5:])
	patches = patches[torch.any(patches != 0, dim=1)]
	#print(patches)
	num_vectors = patches.shape[0]
	if train or iter == 0:
		print(patches.shape)
	
	cosine_similaritys = F.cosine_similarity(patches.unsqueeze(1), patches.unsqueeze(0), dim=2).to(patches.device)
	patches = patches.unsqueeze(1)
	#cosine_similarity = cosine_similaritys.view(-1).cpu().detach()
	#print(cosine_similaritys.shape, cosine_similarity.shape)
	#threshold = np.percentile(cosine_similarity, 100 - threshold_ratio)
	#threshold = 0.995
	#print("threshold:", threshold)
	#print(tensor.device)
	#print("tensor:", tensor.shape)
	
	# 用来存储每个向量的类别
	labels = -1 * torch.ones(num_vectors, dtype=torch.long).to(patches.device)
	current_label = 0
	not_similarity = []
	similarity = []
	word_freqs = []
	word_num = 0
	word_kinds = []
	
	for i in range(num_vectors):
		if labels[i] == -1:  # 该向量尚未分类
			#labels[i] = current_label
			idx = (cosine_similaritys[i] >= threshold) & (labels == -1)
			#print(idx)
			# 将其标记为当前类别
			labels[idx] = current_label
			#print("labels[idx]:", labels[idx].shape)
			word_freq = cosine_similaritys[i][idx].sum()
			#print(labels[idx].shape, word_freq)
			word_freqs.append(word_freq)
			word_num += word_freq
			#word_kinds.append(cosine_similaritys[i][idx].mean())
			current_label += 1  # 移动到下一个类别
		#idx = cosine_similaritys[i] >= threshold
		##print(idx)
		#word_freq = cosine_similaritys[i][idx].sum()
		##print(labels[idx].shape, word_freq)
		#word_freqs.append(word_freq)
		#word_num += word_freq
		#word_kinds.append(cosine_similaritys[i][idx].mean())
	
	# 
	#class_counts = torch.tensor([torch.sum(labels == i).item() for i in range(current_label)]).to(patches.device)
	#print(class_counts)
	#class_counts = class_counts.float().mean()
	#print("current_label:", current_label, "class_counts:", class_counts.item())
	#print("current_label:", current_label)
	#for w in word_freqs:
	#	print(w.item())

	word_freqs = torch.tensor(word_freqs)
	#print(word_freqs)
	#word_kinds = torch.tensor(word_kinds)
	word_freq = torch.mean(word_freqs)
	#print(word_kinds[word_freqs >= word_freq])
	#word_kind = word_kinds[word_freqs >= word_freq].sum()
	if train or iter == 0:
		print("word_freq:", word_freq.item(), "word_num:", word_num.item())
	aux_loss = torch.log(word_num / word_freq) 
	#print("word_freq:", word_freq.item(), "num_vectors:", num_vectors)
	#aux_loss = torch.log(num_vectors / word_freq) #+ torch.log(word_num / word_kind)
	
	
	return aux_loss

def calcute_aux_loss0(patches, threshold, train, iter, high_freq=10, model_verbose=0):
	# if train or iter == 0:
	# 	print(patches.shape, end=' ')
	#print(patches[-5:])
	# patches = patches[torch.any(patches != 0, dim=1)]
	#print(patches)
	num_vectors = patches.shape[0]
	if train or iter == 0:
		if model_verbose:
			print(patches.shape)
	
	# Normalize the patches to compute cosine similarity via dot product
	patches_norm = F.normalize(patches, dim=1)
	# similarity_fn = lambda x, y: torch.matmul(x, y.T)  # dot product == cosine since already normalized
	# cosine_similarities = similarity_fn(patches_norm, patches_norm)  # [N, N]
	with torch.no_grad():
		cosine_similarities = torch.matmul(patches_norm, patches_norm.T)
		torch.diagonal(cosine_similarities).fill_(1.0)
	# similarity_mask = cosine_similarities >= threshold  # boolean mask
	# cosine_similaritys = F.cosine_similarity(patches.unsqueeze(1), patches.unsqueeze(0), dim=2).to(patches.device)
	# 归一化
	# print(cosine_similarities)
	#cosine_similaritys = 0.5 + 0.5 * cosine_similaritys
	#print(cosine_similaritys)
	patches = patches.unsqueeze(1)
	
	# 用来存储每个向量的类别
	labels = -1 * torch.ones(num_vectors, dtype=torch.long).to(patches.device)
	# 创建相似度掩码（>=阈值）
	mask = cosine_similarities >= threshold
	#cosine_similaritys = cosine_similaritys.masked_fill(~mask, 0.0)
	# 生成未分类掩码
	unlabeled_mask = (labels == -1)
	# 找到所有未分类点的索引
	# unlabeled_indices = torch.where(unlabeled_mask)[0]
	current_label = 0
	word_freqs = 0.
	word_num = 0.
	word_kinds = []

	while unlabeled_mask.any():
		# 获取未分类点的下标
		# unlabeled_indices = torch.where(unlabeled_mask)[0]
		unlabeled_indices = torch.nonzero(unlabeled_mask, as_tuple=False).view(-1)
		# print("unlabeled_indices:", unlabeled_indices.shape)
		#unlabeled_mask_2d = unlabeled_mask.unsqueeze(0) & unlabeled_mask.unsqueeze(1)
		#print("unlabeled_mask_2d:", unlabeled_mask_2d)
	
		# 计算每个未分类点的"度"（相似连接数）
		#print("mask:", mask.shape)
		#print("mask:",mask)
		degrees = mask[unlabeled_mask][:, unlabeled_mask].sum(dim=1)
		# print("degrees:", degrees.shape)
		# print("degrees:", degrees)
	
		# Find center with highest degree
		max_degree_idx = torch.argmax(degrees)
		# print("max_degree_idx:", max_degree_idx)
		center_idx = unlabeled_indices[max_degree_idx]
		# print("center_idx:", center_idx)
	
		# Get similar and unlabeled indices
		similar_mask = mask[center_idx] & unlabeled_mask
		# num = similar_mask.sum()
		# print("num:", num)
		# if num < high_freq * num_vectors:
		# 	# word_num += unlabeled_mask.sum()
		# 	# print(unlabeled_mask.sum())
		# 	break
	
		# 标记这些点为当前类别
		labels[similar_mask] = current_label
		# 更新未分类掩码
		# unlabeled_mask = (labels == -1)
		unlabeled_mask[similar_mask] = False
		current_label += 1
	
		# 计算当前类别的词频（相似度之和）
		# word_freq = cosine_similarities[center_idx, similar_mask].sum()
		# print(word_freq)
		# print(patches_norm[center_idx].shape)
		similar_mask[center_idx] = False
		word_freq = 0.
		if similar_mask.any():
			word_freq += torch.matmul(patches_norm[similar_mask] , patches_norm[center_idx]).sum()
		word_freq += 1.0
		# print(word_freq)
		word_freqs += word_freq
		# word_num += word_freq
		#word_kinds.append(cosine_similaritys[center_idx, similar_mask].mean())
		
		if current_label > num_vectors:
			raise RuntimeError("循环超过最大次数，可能进入死循环")
		if current_label > high_freq:
			break

	# word_freqs = torch.tensor(word_freqs)
	#print(word_freqs)
	word_kind = torch.tensor(word_kinds).sum()
	# word_freq = torch.mean(word_freqs)
	word_num = torch.tensor(num_vectors)
	word_freq = word_num
	if current_label != 0:
		word_freq = word_freqs / current_label
	
	#print(word_kinds[word_freqs >= word_freq])
	#word_kind = word_kinds
	if train or iter == 0:
		if model_verbose:
			if len(word_kinds) > 0:
				print("word_freq:", word_freq.item(), "word_num:", word_num.item(), "word_kind:", word_kind.item())
			else:
				print("word_freq:", word_freq.item(), "word_num:", word_num.item())
	aux_loss = torch.log(word_num / word_freq) 
	# if len(word_kinds) > 0:
	# 	aux_loss += torch.log(word_num / word_kind) 
	
	return aux_loss

def calcute_aux_loss(patches, threshold, train, iter, high_freq=1.5, model_verbose=0):
	num_vectors = patches.shape[0]
	if train or iter == 0:
		if model_verbose:
			print(patches.shape)
	
	# Normalize the patches to compute cosine similarity via dot product
	patches_norm = F.normalize(patches, dim=1)
	with torch.no_grad():
		cosine_similarities = torch.matmul(patches_norm, patches_norm.T)
		torch.diagonal(cosine_similarities).fill_(1.0)
		# 用来存储每个向量的类别
		labels = -1 * torch.ones(num_vectors, dtype=torch.long).to(patches.device)
		# 创建相似度掩码（>=阈值）
		mask = cosine_similarities >= threshold
		#cosine_similaritys = cosine_similaritys.masked_fill(~mask, 0.0)
		# 生成未分类掩码
		unlabeled_mask = (labels == -1)
		# 找到所有未分类点的索引
		# unlabeled_indices = torch.where(unlabeled_mask)[0]
		current_label = 0
		word_freqs = []

		while unlabeled_mask.any():
			# 获取未分类点的下标
			unlabeled_indices = torch.nonzero(unlabeled_mask, as_tuple=False).view(-1)
			# print("unlabeled_indices:", unlabeled_indices.shape)
			degrees = mask[unlabeled_mask][:, unlabeled_mask].sum(dim=1)
			# print("degrees:", degrees.shape)
			# print("degrees:", degrees)
			# Find center with highest degree
			max_degree_idx = torch.argmax(degrees)
			# print("max_degree_idx:", max_degree_idx)
			center_idx = unlabeled_indices[max_degree_idx]
			# print("center_idx:", center_idx)
			# Get similar and unlabeled indices
			similar_mask = mask[center_idx] & unlabeled_mask
			# 标记这些点为当前类别
			labels[similar_mask] = current_label
			# 更新未分类掩码
			# unlabeled_mask = (labels == -1)
			unlabeled_mask[similar_mask] = False
			current_label += 1
			# 计算当前类别的词频（相似度之和）
			word_freq = similar_mask.sum()
			# print(word_freq)
			word_freqs.append(word_freq)
		# 计算频率均值和标准差
		word_freqs = torch.tensor(word_freqs).float()
		mean = torch.mean(word_freqs)
		std = 0.
		if len(word_freqs) > 1:
			std = torch.std(word_freqs)
		# print(word_freqs.shape, mean, std)
		# 阈值设为均值 + k × 标准差
		high_freq_threshold = mean + high_freq * std

	# 用来存储每个向量的类别
	labels = -1 * torch.ones(num_vectors, dtype=torch.long).to(patches.device)
	# 创建相似度掩码（>=阈值）
	mask = cosine_similarities >= threshold
	#cosine_similaritys = cosine_similaritys.masked_fill(~mask, 0.0)
	# 生成未分类掩码
	unlabeled_mask = (labels == -1)
	# 找到所有未分类点的索引
	# unlabeled_indices = torch.where(unlabeled_mask)[0]
	current_label = 0
	word_freqs = 0.
	word_num = 0.
	word_kinds = []

	while unlabeled_mask.any():
		# 获取未分类点的下标
		# unlabeled_indices = torch.where(unlabeled_mask)[0]
		unlabeled_indices = torch.nonzero(unlabeled_mask, as_tuple=False).view(-1)
		# print("unlabeled_indices:", unlabeled_indices.shape)
	
		# 计算每个未分类点的"度"（相似连接数）
		#print("mask:", mask.shape)
		#print("mask:",mask)
		degrees = mask[unlabeled_mask][:, unlabeled_mask].sum(dim=1)
		# print("degrees:", degrees.shape)
		# print("degrees:", degrees)
	
		# Find center with highest degree
		max_degree_idx = torch.argmax(degrees)
		# print("max_degree_idx:", max_degree_idx)
		center_idx = unlabeled_indices[max_degree_idx]
		# print("center_idx:", center_idx)
	
		# Get similar and unlabeled indices
		similar_mask = mask[center_idx] & unlabeled_mask
		num = similar_mask.sum()
		# print("num:", num)
		if num < high_freq_threshold:
			# word_num += unlabeled_mask.sum()
			# print(unlabeled_mask.sum())
			break
	
		# 标记这些点为当前类别
		labels[similar_mask] = current_label
		# 更新未分类掩码
		# unlabeled_mask = (labels == -1)
		unlabeled_mask[similar_mask] = False
		current_label += 1
	
		# 计算当前类别的词频（相似度之和）
		# word_freq = cosine_similarities[center_idx, similar_mask].sum()
		# print(word_freq)
		# print(patches_norm[center_idx].shape)
		similar_mask[center_idx] = False
		word_freq = 0.
		if similar_mask.any():
			word_freq += torch.matmul(patches_norm[similar_mask] , patches_norm[center_idx]).sum()
		word_freq += 1.0
		# print(word_freq)
		word_freqs += word_freq
		# word_num += word_freq
		#word_kinds.append(cosine_similaritys[center_idx, similar_mask].mean())
		
		if current_label > num_vectors:
			raise RuntimeError("循环超过最大次数，可能进入死循环")

	# word_freqs = torch.tensor(word_freqs)
	#print(word_freqs)
	word_kind = torch.tensor(word_kinds).sum()
	# word_freq = torch.mean(word_freqs)
	word_num = torch.tensor(num_vectors)
	word_freq = word_num
	if current_label != 0:
		word_freq = word_freqs / current_label
	# high_freq_threshold = torch.tensor(high_freq_threshold)
	
	#print(word_kinds[word_freqs >= word_freq])
	#word_kind = word_kinds
	aux_loss = torch.log(word_num / word_freq) 
	if train or iter == 0:
		if model_verbose:
			if len(word_kinds) > 0:
				print("word_freq:", to_scalar(word_freq), "word_num:", to_scalar(word_num), "word_kind:", to_scalar(word_kind))
			else:
				print("word_freq:", to_scalar(word_freq), "word_num:", to_scalar(word_num), "high_freq_threshold:", to_scalar(high_freq_threshold))
	# if len(word_kinds) > 0:
	# 	aux_loss += torch.log(word_num / word_kind) 
	
	return aux_loss

def to_scalar(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().item()
    if isinstance(x, np.generic):
        return x.item()
    return x

def calcute_aux_loss_cos(patches, threshold, train, iter):
	# if train or iter == 0:
	# 	print(patches.shape, end=' ')
	#print(patches[-5:])
	# patches = patches[torch.any(patches != 0, dim=1)]
	#print(patches)
	num_vectors = patches.shape[0]
	if train or iter == 0:
		print(patches.shape)
	
	# Normalize the patches to compute cosine similarity via dot product
	patches_norm = F.normalize(patches, dim=1)
	similarity_fn = lambda x, y: torch.matmul(x, y.T)  # dot product == cosine since already normalized
	cosine_similarities = similarity_fn(patches_norm, patches_norm)  # [N, N]
	# similarity_mask = cosine_similarities >= threshold  # boolean mask
	# cosine_similaritys = F.cosine_similarity(patches.unsqueeze(1), patches.unsqueeze(0), dim=2).to(patches.device)
	# 归一化
	#print(cosine_similaritys)
	#cosine_similaritys = 0.5 + 0.5 * cosine_similaritys
	#print(cosine_similaritys)
	patches = patches.unsqueeze(1)
	
	# 用来存储每个向量的类别
	labels = -1 * torch.ones(num_vectors, dtype=torch.long).to(patches.device)
	# 创建相似度掩码（>=阈值）
	mask = cosine_similarities >= threshold
	#cosine_similaritys = cosine_similaritys.masked_fill(~mask, 0.0)
	# 生成未分类掩码
	unlabeled_mask = (labels == -1)
	# 找到所有未分类点的索引
	unlabeled_indices = torch.where(unlabeled_mask)[0]
	current_label = 0
	word_freqs = []
	word_num = 0
	word_kinds = []
	
	#while unlabeled_indices.numel() > 0:
	#	# 取第一个未分类点作为当前中心
	#	i = unlabeled_indices[0]
	#	# 找出所有与当前中心相似且未分类的点
	#	idx = mask[i] & unlabeled_mask
	#	# 标记这些点为当前类别
	#	labels[idx] = current_label
	#	# 更新未分类掩码
	#	unlabeled_mask = (labels == -1)
	
	#	# 计算当前类别的词频（相似度之和）
	#	word_freq = cosine_similaritys[i, idx].sum()
	#	word_freqs.append(word_freq)
	#	word_num += word_freq
	#	#word_kinds.append(cosine_similaritys[i, idx].mean())
	
	#	current_label += 1
	#	unlabeled_indices = torch.where(unlabeled_mask)[0]

	while unlabeled_mask.any():
		# 获取未分类点的下标
		unlabeled_indices = torch.where(unlabeled_mask)[0]
		#print("unlabeled_indices:", unlabeled_indices)
		#unlabeled_mask_2d = unlabeled_mask.unsqueeze(0) & unlabeled_mask.unsqueeze(1)
		#print("unlabeled_mask_2d:", unlabeled_mask_2d)
	
		# 计算每个未分类点的"度"（相似连接数）
		#print("mask:", mask.shape)
		#print("mask:",mask)
		degrees = mask[unlabeled_mask][:, unlabeled_mask].sum(dim=1)
		#print("degrees:", degrees.shape)
		#print("degrees:", degrees)
	
		# 找到度最高的未分类点
		max_degree_idx = torch.argmax(degrees)
		#print("max_degree_idx:", max_degree_idx)
		center_idx = unlabeled_indices[max_degree_idx]
		#print("center_idx:", center_idx)
	
		# 找出所有与当前中心相似且未分类的点
		similar_mask = mask[center_idx] & unlabeled_mask
	
		# 标记这些点为当前类别
		labels[similar_mask] = current_label
	
		# 计算当前类别的词频（相似度之和）
		word_freq = cosine_similarities[center_idx, similar_mask].sum()
		word_freqs.append(word_freq)
		word_num += word_freq
		#word_kinds.append(cosine_similaritys[center_idx, similar_mask].mean())
	
		# 更新未分类掩码
		unlabeled_mask = (labels == -1)
		current_label += 1

	word_freqs = torch.tensor(word_freqs)
	#print(word_freqs)
	word_kind = torch.tensor(word_kinds).sum()
	word_freq = torch.mean(word_freqs)
	#print(word_kinds[word_freqs >= word_freq])
	#word_kind = word_kinds
	if train or iter == 0:
		if len(word_kinds) > 0:
			print("word_freq:", word_freq.item(), "word_num:", word_num.item(), "word_kind:", word_kind.item())
		else:
			print("word_freq:", word_freq.item(), "word_num:", word_num.item())
	aux_loss = torch.log(word_num / word_freq) 
	if len(word_kinds) > 0:
		aux_loss += torch.log(word_num / word_kind) 
	
	return aux_loss

def calcute_aux_loss_in(patches, threshold, train, iter):
	if train or iter == 0:
		print(patches.shape, end=' ')
	#print(patches[-5:])
	patches = patches[torch.any(patches != 0, dim=1)]
	#print(patches)
	num_vectors = patches.shape[0]
	if train or iter == 0:
		print(patches.shape)
	
	dists  = torch.cdist(patches, patches, p=2)
	# 归一化
	#print(dists)
	dists = 1 / (1 + dists)
	#print("dist:", dists.shape)
	#print("dist:", dists)
	#cosine_similaritys = F.cosine_similarity(patches.unsqueeze(1), patches.unsqueeze(0), dim=2).to(patches.device)
	patches = patches.unsqueeze(1)
	
	# 用来存储每个向量的类别
	labels = -1 * torch.ones(num_vectors, dtype=torch.long).to(patches.device)
	# 创建相似度掩码（>=阈值）
	mask = dists >= threshold
	dists = dists.masked_fill(~mask, 0.0)
	# 生成未分类掩码
	unlabeled_mask = (labels == -1)
	# 找到所有未分类点的索引
	unlabeled_indices = torch.where(unlabeled_mask)[0]
	current_label = 0
	word_freqs = []
	word_num = 0
	word_kinds = []

	while unlabeled_mask.any():
		# 获取未分类点的下标
		unlabeled_indices = torch.where(unlabeled_mask)[0]
	
		# 计算每个未分类点的"度"（相似连接数）
		#print("mask:", mask.shape)
		#print("mask:",mask)
		degrees = mask[unlabeled_mask][:, unlabeled_mask].sum(dim=1)
		#print("degrees:", degrees.shape)
		#print("degrees:", degrees)
	
		# 找到度最高的未分类点
		max_degree_idx = torch.argmax(degrees)
		#print("max_degree_idx:", max_degree_idx)
		center_idx = unlabeled_indices[max_degree_idx]
		#print("center_idx:", center_idx)
	
		# 找出所有与当前中心相似且未分类的点
		similar_mask = mask[center_idx] & unlabeled_mask
	
		# 标记这些点为当前类别
		labels[similar_mask] = current_label
	
		# 计算当前类别的词频（相似度之和）
		#word_freq = (threshold - dists[center_idx, similar_mask]).sum()
		word_freq = dists[center_idx, similar_mask].sum()
		word_freqs.append(word_freq)
		word_num += word_freq
		#word_kinds.append(cosine_similaritys[center_idx, similar_mask].mean())
	
		# 更新未分类掩码
		unlabeled_mask = (labels == -1)
		current_label += 1

	word_freqs = torch.tensor(word_freqs)
	#print(word_freqs)
	word_kind = torch.tensor(word_kinds).sum()
	word_freq = torch.mean(word_freqs)
	#print(word_kinds[word_freqs >= word_freq])
	#word_kind = word_kinds
	if train or iter == 0:
		if len(word_kinds) > 0:
			print("word_freq:", word_freq.item(), "word_num:", word_num.item(), "word_kind:", word_kind.item())
		else:
			print("word_freq:", word_freq.item(), "word_num:", word_num.item())
	aux_loss = torch.log(word_num / word_freq) 
	#aux_loss = word_freq / word_num 
	if len(word_kinds) > 0:
		aux_loss += torch.log(word_num / word_kind) 
	
	return aux_loss

def calcute_aux_loss_in(patches, threshold, train, iter):
	if train or iter == 0:
		print(patches.shape, end=' ')
	#print(patches[-5:])
	patches = patches[torch.any(patches != 0, dim=1)]
	#print(patches)
	num_vectors = patches.shape[0]
	if train or iter == 0:
		print(patches.shape)
	
	# ||x - y||^2 = ||x||^2 + ||y||^2 - 2⟨x, y⟩
	x = patches
	x_square = (x ** 2).sum(dim=1, keepdim=True)
	#print("x_square:", x_square.shape)
	#print("x_square:", x_square)
	# 计算 pairwise distance 矩阵
	dist = x_square + x_square.T - 2 * x @ x.T
	#print("dist:", dist.shape)
	#print("dist:", dist)
	# 由于数值误差，有可能出现负值，取 max 防止 sqrt 负数
	dists = torch.sqrt(torch.clamp(dist, min=0.0))
	#print("dist:", dists.shape)
	#print("dist:", dists)
	#cosine_similaritys = F.cosine_similarity(patches.unsqueeze(1), patches.unsqueeze(0), dim=2).to(patches.device)
	patches = patches.unsqueeze(1)
	
	# 用来存储每个向量的类别
	labels = -1 * torch.ones(num_vectors, dtype=torch.long).to(patches.device)
	# 创建相似度掩码（<=阈值）
	mask = dists <= threshold
	dists = dists.masked_fill(~mask, 0.0)
	# 生成未分类掩码
	unlabeled_mask = (labels == -1)
	# 找到所有未分类点的索引
	unlabeled_indices = torch.where(unlabeled_mask)[0]
	current_label = 0
	word_freqs = []
	word_num = 0
	word_kinds = []

	while unlabeled_mask.any():
		# 获取未分类点的下标
		unlabeled_indices = torch.where(unlabeled_mask)[0]
	
		# 计算每个未分类点的"度"（相似连接数）
		#print("mask:", mask.shape)
		#print("mask:",mask)
		degrees = mask[unlabeled_mask][:, unlabeled_mask].sum(dim=1)
		#print("degrees:", degrees.shape)
		#print("degrees:", degrees)
	
		# 找到度最高的未分类点
		max_degree_idx = torch.argmax(degrees)
		#print("max_degree_idx:", max_degree_idx)
		center_idx = unlabeled_indices[max_degree_idx]
		#print("center_idx:", center_idx)
	
		# 找出所有与当前中心相似且未分类的点
		similar_mask = mask[center_idx] & unlabeled_mask
	
		# 标记这些点为当前类别
		labels[similar_mask] = current_label
	
		# 计算当前类别的词频（相似度之和）
		word_freq = dists[center_idx, similar_mask].sum()
		word_freqs.append(word_freq)
		word_num += word_freq
		#word_kinds.append(cosine_similaritys[center_idx, similar_mask].mean())
	
		# 更新未分类掩码
		unlabeled_mask = (labels == -1)
		current_label += 1

	word_freqs = torch.tensor(word_freqs)
	#print(word_freqs)
	word_kind = torch.tensor(word_kinds).sum()
	word_freq = torch.mean(word_freqs)
	#print(word_kinds[word_freqs >= word_freq])
	#word_kind = word_kinds
	if train or iter == 0:
		if len(word_kinds) > 0:
			print("word_freq:", word_freq.item(), "word_num:", word_num.item(), "word_kind:", word_kind.item())
		else:
			print("word_freq:", word_freq.item(), "word_num:", word_num.item())
	#aux_loss = torch.log(word_num / word_freq) 
	aux_loss = word_freq / word_num 
	if len(word_kinds) > 0:
		aux_loss += torch.log(word_num / word_kind) 
	
	return aux_loss

def calcute_aux_loss0(patches, threshold, train, iter):
	if train or iter == 0:
		print(patches.shape, end=' ')
	#print(patches[-5:])
	patches = patches[torch.any(patches != 0, dim=1)]
	num_vectors = patches.shape[0]
	if train or iter == 0:
		print(patches.shape)
	
	cosine_similaritys = F.cosine_similarity(patches.unsqueeze(1), patches.unsqueeze(0), dim=2).to(patches.device)
	patches = patches.unsqueeze(1)

	# 提取上三角部分
	upper_triangle = torch.triu(cosine_similaritys)

	# 应用阈值
	upper_triangle[upper_triangle < threshold] = 0
	#print(upper_triangle)

	## 非0元素的位置
	#nonzero_mask = upper_triangle != 0
	## 沿列累加后，第一次出现非零的地方为1，其它位置大于1
	#cumsum_mask = nonzero_mask.cumsum(dim=0)
	#print(cumsum_mask)
	## 获取对角线元素
	#diag = torch.diagonal(cumsum_mask)
	## 找到对角线上>1的索引
	#zero_diag_mask = (diag > 1)
	## 创建行掩码，形状为 (n, 1)，用于广播
	#row_mask = ~zero_diag_mask[:, None]  # 反转
	## 用行掩码将对应行变为 0
	#upper_triangle = upper_triangle * row_mask
	
	## 用 first_nonzero_mask 进行掩码
	#temp = upper_triangle * first_nonzero_mask
	##print(upper_triangle[:5])
	##print(upper_triangle[-1])
	## 获取对角线元素
	#diag = torch.diagonal(temp)
	##print("diag:", diag)
	## 找到对角线上为0的索引（i 行的 i 列元素为 0）
	#zero_diag_mask = (diag == 0)
	## 创建行掩码，形状为 (n, 1)，用于广播
	#row_mask = ~zero_diag_mask[:, None]  # 反转

	# 非0元素的位置
	nonzero_mask = upper_triangle != 0
	#print(nonzero_mask)
	nonzero_mask = torch.flip(nonzero_mask, dims=[0])
	#nonzero_mask = torch.tensor(nonzero_mask)
	# 沿列累加后，第一次出现非零的地方为1，其它位置大于1
	cumsum_mask = nonzero_mask.cumsum(dim=0)
	#print(cumsum_mask)
	# 只保留每列第一次非零的位置（即 cumsum_mask==1 的位置）
	first_nonzero_mask = cumsum_mask == 1
	first_nonzero_mask = torch.flip(first_nonzero_mask, dims=[0])
	#print(first_nonzero_mask)
	upper_triangle = upper_triangle * first_nonzero_mask

	# 对每一行求和
	word_freqs = torch.sum(upper_triangle, dim=1)
	word_freqs = word_freqs[word_freqs != 0]
	#print(word_freqs.shape)
	print(word_freqs)
	word_freq = torch.mean(word_freqs)
	word_num = torch.sum(word_freqs)
	if train or iter == 0:
		print("word_freq:", word_freq.item(), "word_num:", word_num.item())
	aux_loss = torch.log(word_num / word_freq) 
	
	return aux_loss

#	num_vectors = patches.shape[0]
	
#	cosine_similaritys = F.cosine_similarity(patches.unsqueeze(1), patches.unsqueeze(0), dim=2).to(patches.device)
#	patches = patches.unsqueeze(1)
	
#	# 用来存储每个向量的类别
#	labels = -1 * torch.ones(num_vectors, dtype=torch.long).to(patches.device)
#	current_label = 0
#	word_freqs = []
#	word_num = 0
#	word_kinds = []

#for i in range(num_vectors):
#		if labels[i] == -1:  # 该向量尚未分类
#			idx = (cosine_similaritys[i] >= threshold) & (labels == -1)
#			# 将其标记为当前类别
#			labels[idx] = current_label
#			word_freq = cosine_similaritys[i][idx].sum()
#			word_freqs.append(word_freq)
#			word_num += word_freq
#			word_kinds.append(cosine_similaritys[i][idx].mean())
#			current_label += 1  # 移动到下一个类别