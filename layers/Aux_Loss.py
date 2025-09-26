import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Semantic Tokenization Regularization (STR)
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
		# Stores the category of each vector
		labels = -1 * torch.ones(num_vectors, dtype=torch.long).to(patches.device)
		# Create a similarity mask (≥ threshold)
		mask = cosine_similarities >= threshold
		#cosine_similaritys = cosine_similaritys.masked_fill(~mask, 0.0)
		# Generate an unclassified mask
		unlabeled_mask = (labels == -1)
		# Find the indices of all unclassified points
		# unlabeled_indices = torch.where(unlabeled_mask)[0]
		current_label = 0
		word_freqs = []

		while unlabeled_mask.any():
			# Get the indices of unclassified points
			unlabeled_indices = torch.nonzero(unlabeled_mask, as_tuple=False).view(-1)
			degrees = mask[unlabeled_mask][:, unlabeled_mask].sum(dim=1)
			# Find center with highest degree
			max_degree_idx = torch.argmax(degrees)
			center_idx = unlabeled_indices[max_degree_idx]
			# Get similar and unlabeled indices
			similar_mask = mask[center_idx] & unlabeled_mask
			# Label these points with the current category
			labels[similar_mask] = current_label
			# Update the unclassified mask
			# unlabeled_mask = (labels == -1)
			unlabeled_mask[similar_mask] = False
			current_label += 1
			# Compute the token frequency for the current class (sum of similarities)
			word_freq = similar_mask.sum()
			# print(word_freq)
			word_freqs.append(word_freq)
		# Compute the mean and standard deviation of the frequencies
		word_freqs = torch.tensor(word_freqs).float()
		mean = torch.mean(word_freqs)
		std = 0.
		if len(word_freqs) > 1:
			std = torch.std(word_freqs)
		# print(word_freqs.shape, mean, std)
		# Set the threshold as the mean + k * the standard deviation
		high_freq_threshold = mean + high_freq * std

	labels = -1 * torch.ones(num_vectors, dtype=torch.long).to(patches.device)
	mask = cosine_similarities >= threshold
	#cosine_similaritys = cosine_similaritys.masked_fill(~mask, 0.0)
	unlabeled_mask = (labels == -1)
	# unlabeled_indices = torch.where(unlabeled_mask)[0]
	current_label = 0
	word_freqs = 0.
	word_num = 0.
	word_kinds = []

	while unlabeled_mask.any():
		# unlabeled_indices = torch.where(unlabeled_mask)[0]
		unlabeled_indices = torch.nonzero(unlabeled_mask, as_tuple=False).view(-1)
		# print("unlabeled_indices:", unlabeled_indices.shape)
	
		# Compute the degree (number of similarity connections) for each unclassified point
		degrees = mask[unlabeled_mask][:, unlabeled_mask].sum(dim=1)
	
		# Find center with highest degree
		max_degree_idx = torch.argmax(degrees)
		center_idx = unlabeled_indices[max_degree_idx]
	
		# Get similar and unlabeled indices
		similar_mask = mask[center_idx] & unlabeled_mask
		num = similar_mask.sum()
		if num < high_freq_threshold:
			# word_num += unlabeled_mask.sum()
			# print(unlabeled_mask.sum())
			break
	
		labels[similar_mask] = current_label
		# unlabeled_mask = (labels == -1)
		unlabeled_mask[similar_mask] = False
		current_label += 1
	
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


