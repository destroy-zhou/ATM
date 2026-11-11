"""Patch tokenizer with selectable adaptive or fixed non-overlapping patches.

Adaptive mode uses hard contiguous segments in the forward pass and soft
ordinal memberships for the backward pass. Fixed mode uses non-overlapping
patches of ``patch_len`` points. Adaptive boundaries can be shared by all
channels or predicted independently for each channel.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from ts_benchmark.baselines.atm.layers.StandardNorm import Normalize
from ts_benchmark.baselines.atm.layers.ForecastTokenEncoder import ForecastTokenEncoder


class BoundaryTimeMLP(nn.Module):
    """Temporal mixer with selectable normalization and boundary readout."""

    def __init__(self, seq_len, d_model, point_head=False, post_norm=False,
                 input_dim=1, external_embedding=False, time_hidden_ratio=0.0):
        super().__init__()
        self.point_head = point_head
        self.post_norm = post_norm
        time_hidden_ratio = float(time_hidden_ratio)
        if not math.isfinite(time_hidden_ratio) or time_hidden_ratio < 0:
            raise ValueError('broadcast_time_hidden_ratio must be finite and non-negative')
        # Zero preserves the original boundary-MLP width; a positive value
        # scales it with the (possibly small-patch-reduced) token count.
        time_hidden = (max(8, d_model // 2) if time_hidden_ratio == 0
                       else max(1, int(seq_len * time_hidden_ratio)))
        self.seq_len = seq_len
        self.value_embedding = None if external_embedding else nn.Linear(input_dim, d_model)
        self.time_norm = nn.LayerNorm(d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, time_hidden),
            nn.GELU(),
            nn.Linear(time_hidden, seq_len),
        )
        self.feature_norm = nn.LayerNorm(d_model)
        self.feature_ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.head = nn.Linear(d_model if point_head else 3 * d_model, 1)

    def encode(self, values):
        if values.shape[1] != self.seq_len:
            raise ValueError(
                'boundary_network=time_mlp requires input length to match seq_len'
            )
        h = self.value_embedding(values.unsqueeze(-1) if values.ndim == 2 else values)
        return self.encode_features(h)

    def encode_features(self, h):
        if h.shape[1] != self.seq_len:
            raise ValueError('Boundary token length must match configured length')
        if self.post_norm:
            h = self.time_norm(h + self.time_mlp(h.transpose(1, 2)).transpose(1, 2))
            return self.feature_norm(h + self.feature_ffn(h))
        mixed = self.time_mlp(self.time_norm(h).transpose(1, 2)).transpose(1, 2)
        h = h + mixed
        return h + self.feature_ffn(self.feature_norm(h))

    def score(self, h):
        if self.point_head:
            # Logit t denotes a cut AFTER point t; the final point needs no cut.
            return self.head(h[:, :-1]).squeeze(-1)
        gap_features = torch.cat(
            (h[:, :-1], h[:, 1:], h[:, 1:] - h[:, :-1]), dim=-1
        )
        return self.head(gap_features).squeeze(-1)


class CompactTokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.bins = config.patch_bins
        self.patch_pooling = getattr(config, 'patch_pooling', 'bins').lower()
        self.patch_query_use_kv_projection = bool(
            getattr(config, 'patch_query_use_kv_projection', False)
        )
        self.patch_mode = getattr(config, 'patch_mode', 'fixed').lower()
        self.patch_len = getattr(config, 'patch_len', 16)
        self.min_len = config.patch_min_len
        self.max_len = config.patch_max_len
        self.threshold = config.boundary_threshold
        self.temperature = config.boundary_temperature
        self.boundary_ste_scale = getattr(config, 'boundary_ste_scale', 0.1)
        self.use_ordinal_st = getattr(config, 'use_ordinal_st', True)
        self.boundary_network = getattr(config, 'boundary_network', 'conv').lower()
        self.merge_weight = float(getattr(config, 'boundary_merge_weight', 0.0))
        self.merge_mode = getattr(config, 'boundary_merge_mode', 'dense').lower()
        self.merge_window = int(getattr(config, 'boundary_merge_window', 8))
        self.merge_threshold = float(getattr(config, 'boundary_merge_threshold', 0.1))
        self.merge_temperature = float(getattr(config, 'boundary_merge_temperature', 0.03))
        self.semantic_weight = float(getattr(config, 'boundary_semantic_weight', 0.0))
        self.semantic_window = int(getattr(config, 'boundary_semantic_window', 8))
        self.semantic_threshold = float(getattr(config, 'boundary_semantic_threshold', 0.1))
        self.semantic_temperature = float(getattr(config, 'boundary_semantic_temperature', 0.03))
        self.anchor_weight = float(getattr(config, 'patch_anchor_weight', 0.0))
        self.anchor_count = int(getattr(config, 'patch_anchor_count', 128))
        self.anchor_pair_count = int(getattr(config, 'patch_anchor_pair_count', 0))
        self.anchor_threshold = float(getattr(config, 'patch_anchor_threshold', 0.95))
        self.anchor_variant = getattr(config, 'patch_anchor_variant', 'global').lower()
        self.reconstruction_weight = float(getattr(config, 'patch_reconstruction_weight', 0.0))
        self.count_weight = float(getattr(config, 'patch_count_weight', 0.0))
        self.broadcast_backbone = getattr(config, 'prediction_backbone', 'token_encoder') == 'patch_broadcast_mlp'
        self.small_patch_len = int(getattr(config, 'small_patch_len', 1))
        self.share_embedding = bool(getattr(config, 'small_patch_share_embedding', False))
        if self.small_patch_len < 1 or self.small_patch_len != getattr(config, 'small_patch_len', 1):
            raise ValueError('small_patch_len must be a positive integer')
        if config.seq_len % self.small_patch_len:
            raise ValueError('seq_len must be divisible by small_patch_len')
        if self.small_patch_len > 1 and not self.broadcast_backbone:
            raise ValueError('small_patch_len > 1 requires patch_broadcast_mlp')
        if (self.small_patch_len > 1 or self.share_embedding) and self.boundary_network != 'time_mlp':
            raise ValueError('Small patch embedding / sharing requires boundary_network=time_mlp')
        self.token_seq_len = config.seq_len // self.small_patch_len
        if self.patch_mode not in ('adaptive', 'adaptive_per_channel', 'fixed'):
            raise ValueError(
                'patch_mode must be adaptive, adaptive_per_channel, or fixed'
            )
        if self.patch_pooling not in ('bins', 'boundary_query'):
            raise ValueError('patch_pooling must be bins or boundary_query')
        if self.patch_len < 1:
            raise ValueError('patch_len must be positive')
        if not (1 <= self.min_len <= self.max_len and self.bins >= 1):
            raise ValueError('Invalid patch length or pooling bin configuration')
        # Public length limits remain in original observations; segment() uses units.
        self.min_len = math.ceil(self.min_len / self.small_patch_len)
        self.max_len = int(self.max_len // self.small_patch_len)
        if self.min_len > self.max_len:
            raise ValueError('No small-patch-aligned length satisfies patch_min_len/max_len')
        if self.patch_mode == 'fixed':
            if self.patch_len % self.small_patch_len:
                raise ValueError('fixed patch_len must be divisible by small_patch_len')
            self.patch_len //= self.small_patch_len
        if not (0 < self.threshold < 1 and self.temperature > 0):
            raise ValueError('Invalid boundary threshold or temperature')
        if self.boundary_ste_scale < 0:
            raise ValueError('boundary_ste_scale must be non-negative')
        if self.boundary_network not in ('conv', 'time_mlp'):
            raise ValueError('boundary_network must be conv or time_mlp')
        if self.merge_window < 2 or self.merge_temperature <= 0:
            raise ValueError('Invalid boundary merge supervision configuration')
        if self.merge_mode not in ('dense', 'sparse_peak', 'sparse_rank', 'waveform_lobe'):
            raise ValueError('boundary_merge_mode must be dense, sparse_peak, sparse_rank or waveform_lobe')
        if self.semantic_weight < 0 or self.semantic_window < 1:
            raise ValueError('Invalid semantic boundary supervision configuration')
        if not (0 <= self.semantic_threshold <= 2 and self.semantic_temperature > 0):
            raise ValueError('Invalid semantic boundary target configuration')
        if self.anchor_weight < 0 or self.anchor_count < 1 or self.anchor_pair_count < 0:
            raise ValueError('Invalid sampled-anchor loss configuration')
        if not (-1 <= self.anchor_threshold <= 1):
            raise ValueError('patch_anchor_threshold must be in [-1, 1]')
        if self.anchor_variant not in (
            'global', 'channel', 'channel_min4', 'channel_length_balanced'
        ):
            raise ValueError(
                'patch_anchor_variant must be global, channel, channel_min4, '
                'or channel_length_balanced'
            )
        # Embed each non-overlapping small patch before variable-length pooling.
        # Context modeling is left to patch aggregation and the encoder.
        self.features = nn.Linear(self.small_patch_len, d)
        self.patch_use_statistics = bool(getattr(config, 'patch_use_statistics', True))
        if self.boundary_network == 'conv':
            self.boundary = nn.Sequential(
                nn.Linear(3*d, d), nn.GELU(), nn.Linear(d, 1)
            )
        else:
            self.boundary = BoundaryTimeMLP(
                self.token_seq_len, d,
                point_head=bool(getattr(config, 'boundary_mlp_point_head', False)),
                post_norm=bool(getattr(config, 'mlp_post_norm', False)),
                input_dim=self.small_patch_len, external_embedding=self.share_embedding,
                time_hidden_ratio=getattr(config, 'broadcast_time_hidden_ratio', 0.0),
            )
        if self.patch_pooling == 'boundary_query':
            self.patch_heads = int(getattr(config, 'n_heads', 1))
            if (self.patch_heads < 1
                    or self.patch_heads != getattr(config, 'n_heads', 1)
                    or d % self.patch_heads):
                raise ValueError(
                    'boundary_query requires a positive n_heads that divides d_model'
                )
            self.patch_head_dim = d // self.patch_heads
            # The d_model-wide query is split into independent attention heads.
            # Its total parameter count and the pooled output width stay fixed.
            self.patch_query = nn.Parameter(torch.randn(d) * 0.02)
            if self.patch_query_use_kv_projection:
                self.patch_key = nn.Linear(d, d, bias=False)
                self.patch_value = nn.Linear(d, d)
            self.patch_relative_position = nn.Linear(1, d)
            pooled_dim = d
        else:
            pooled_dim = self.bins * d
        self.project = nn.Linear(pooled_dim + (3 if self.patch_use_statistics else 0), d)
        self.position = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d))
        self.reconstruct = nn.Linear(d, self.bins * self.small_patch_len)

    def boundary_query_pool(self, features, membership, relative):
        """Pool points with a learned query, constrained by patch membership.

        Multiplication followed by renormalization is the stable form of
        adding ``log(membership)`` to attention logits. In the forward pass,
        hard-zero membership gives patch-external points exactly zero weight.
        If ordinal STE is enabled, its surrogate membership still supplies a
        backward path to the boundary probabilities.
        """
        local_position = 2 * relative - 1
        # Expand the linear positional projection algebraically instead of
        # materializing [N, T, K, D], which is expensive when min_len=1.
        if self.patch_query_use_kv_projection:
            keys = self.patch_key(features)
            values = self.patch_value(features)
        else:
            keys = values = features
        n, time_steps, _ = keys.shape
        keys = keys.reshape(n, time_steps, self.patch_heads, self.patch_head_dim)
        values = values.reshape(n, time_steps, self.patch_heads, self.patch_head_dim)
        query = self.patch_query.reshape(self.patch_heads, self.patch_head_dim)
        position_weight = self.patch_relative_position.weight[:, 0].reshape(
            self.patch_heads, self.patch_head_dim
        )
        position_embedding_bias = self.patch_relative_position.bias.reshape(
            self.patch_heads, self.patch_head_dim
        )
        content_logits = torch.einsum('nthd,hd->nth', keys, query)
        position_scale = (position_weight * query).sum(-1)
        position_bias = (position_embedding_bias * query).sum(-1)
        logits = (
            content_logits.unsqueeze(-1)
            + local_position.unsqueeze(2) * position_scale[None, None, :, None]
            + position_bias[None, None, :, None]
        ) / (self.patch_head_dim ** 0.5)
        # Stabilize with the maximum *inside each hard patch*. A global
        # softmax before applying membership can underflow every in-patch
        # weight when an excluded point has a much larger logit. Clamping
        # only affects excluded points above the in-patch maximum and keeps
        # their surrogate membership gradient bounded by one.
        hard_inside = membership.detach().gt(0.5).unsqueeze(2)
        inside_max = logits.masked_fill(
            ~hard_inside, float('-inf')
        ).amax(dim=1, keepdim=True)
        inside_max = torch.where(
            torch.isfinite(inside_max), inside_max, torch.zeros_like(inside_max)
        )
        stable_logits = (logits - inside_max).clamp(max=0)
        constrained = stable_logits.exp() * membership.unsqueeze(2)
        attention = constrained / constrained.sum(1, keepdim=True).clamp_min(1e-8)
        pooled_features = torch.einsum('nthk,nthd->nkhd', attention, values)
        pooled_position = torch.einsum(
            'nthk,ntk->nkh', attention, local_position
        )
        pooled_position_embedding = (
            pooled_position.unsqueeze(-1) * position_weight[None, None]
            + position_embedding_bias[None, None]
        )
        pooled = (pooled_features + pooled_position_embedding).flatten(2)
        pooled = pooled * membership.sum(1).gt(0).unsqueeze(-1)
        return pooled, attention

    def semantic_boundary_dissimilarity(self, features):
        """Compare pooled feature context on both sides of every time gap."""
        _, length, _ = features.shape
        if length < 2:
            return features.new_zeros(features.shape[0], 0)

        # Prefix sums keep the cost linear in sequence length and independent
        # of the context-window size.
        safe = torch.nan_to_num(
            features.float(), nan=0.0, posinf=1e4, neginf=-1e4
        )
        prefix = F.pad(safe.cumsum(1), (0, 0, 1, 0))
        split = torch.arange(1, length, device=features.device)
        window = math.ceil(self.semantic_window / self.small_patch_len)
        left_start = (split - window).clamp_min(0)
        right_end = (split + window).clamp_max(length)
        left = prefix[:, split] - prefix[:, left_start]
        right = prefix[:, right_end] - prefix[:, split]
        left = left / (split - left_start).to(safe.dtype)[None, :, None]
        right = right / (right_end - split).to(safe.dtype)[None, :, None]
        similarity = F.cosine_similarity(left, right, dim=-1, eps=1e-6)
        return (1 - similarity).clamp(0, 2)

    def _pair_anchor_loss(self, tokens, valid, lengths, channels):
        """One total P-pair budget, with replacement and without self-pairs.

        Sample channels equally, and eligible length buckets equally within a
        channel. Only the sampled endpoints are gathered and normalized.
        Fixed-size integer sorting groups valid indices without nonzero or
        data-dependent host branches. Feature arithmetic is O(P*D).
        """
        rows, slots, dim = tokens.shape
        if self.anchor_variant == 'global':
            group_ids = torch.zeros_like(valid, dtype=torch.long)
            channel_count, buckets = 1, 1
        else:
            if channels is None or channels < 1 or rows % channels:
                raise ValueError('Channel anchor variants require a valid channel count')
            if lengths is None or lengths.shape != valid.shape:
                raise ValueError('Channel anchor variants require patch lengths')
            channel_count = channels
            group_ids = (torch.arange(rows, device=tokens.device) % channels)[:, None].expand_as(valid)
            buckets = 1
            if self.anchor_variant == 'channel_min4':
                valid = valid & lengths.ge(4)
            elif self.anchor_variant == 'channel_length_balanced':
                buckets = 4
                bucket = lengths.ge(2).long() + lengths.ge(4).long() + lengths.ge(8).long()
                group_ids = group_ids * buckets + bucket
        groups = channel_count * buckets
        # Invalid slots sort to a final sentinel group. Tensor sizes depend only
        # on input shapes, avoiding nonzero/bincount's dynamic CUDA output sizes.
        labels = group_ids.flatten().masked_fill(~valid.flatten(), groups)
        flat_indices = labels.argsort(stable=True)
        counts = labels.new_zeros(groups + 1).scatter_add(
            0, labels, torch.ones_like(labels)
        )[:groups]
        eligible = counts.ge(2).reshape(channel_count, buckets)
        weights = eligible.float() / eligible.sum(1, keepdim=True).clamp_min(1)
        # Inverse-CDF sampling avoids multinomial's probability validation sync.
        # An all-empty batch uses a safe dummy group, masked out in the loss.
        weights = weights.flatten()
        fallback = torch.arange(groups, device=tokens.device).eq(0).to(weights.dtype)
        weights = torch.where(eligible.any(), weights, fallback)
        cdf = weights.cumsum(0)
        upper = torch.nextafter(cdf[-1], cdf.new_zeros(()))
        draws = (torch.rand(self.anchor_pair_count, device=tokens.device) * cdf[-1]).clamp(max=upper)
        sampled_groups = torch.searchsorted(cdf, draws, right=True)
        group_sizes = counts[sampled_groups].clamp_min(1)
        starts = counts.cumsum(0) - counts
        # Uniform ordered pairs within the selected group; offset excludes self.
        left_rank = (torch.rand(self.anchor_pair_count, device=tokens.device) * group_sizes).long()
        offset = 1 + (torch.rand(self.anchor_pair_count, device=tokens.device) * (group_sizes - 1)).long()
        right_rank = (left_rank + offset) % group_sizes
        left_idx = flat_indices[starts[sampled_groups] + left_rank]
        right_idx = flat_indices[starts[sampled_groups] + right_rank]
        flat_tokens = tokens.reshape(-1, dim)
        endpoints = torch.stack((flat_tokens[left_idx], flat_tokens[right_idx]), dim=1)
        endpoints = F.normalize(torch.nan_to_num(endpoints.float(), nan=0., posinf=1e4, neginf=-1e4), dim=-1, eps=1e-6)
        similarities = (endpoints[:, 0] * endpoints[:, 1]).sum(-1).clamp(-1, 1)
        positive = similarities.detach().ge(self.anchor_threshold)
        # Keep each sampled group's positive-pair normalization independent.
        sums = similarities.new_zeros(groups).scatter_add(0, sampled_groups, (1 - similarities) * positive)
        positives = similarities.new_zeros(groups).scatter_add(0, sampled_groups, positive.float())
        pair_counts = similarities.new_zeros(groups).scatter_add(0, sampled_groups, torch.ones_like(similarities))
        cohesion = sums / positives.clamp_min(1)
        # Compute variance only from sampled endpoints, separately per group.
        scatter_index = sampled_groups[:, None].expand(-1, dim)
        # Shift before accumulating means, then use centered squared deviations.
        # This avoids subtracting two nearly equal second moments at collapse.
        shifted = endpoints - endpoints[0, 0].detach()
        feature_sum = endpoints.new_zeros(groups, dim).scatter_add(0, scatter_index, shifted.sum(1))
        denominator = (2 * pair_counts).clamp_min(1)[:, None]
        centered = shifted - (feature_sum / denominator)[sampled_groups, None, :]
        square_sum = endpoints.new_zeros(groups, dim).scatter_add(0, scatter_index, centered.square().sum(1))
        variance = square_sum / denominator
        feature_std = (variance + 1e-12).sqrt()
        anti_collapse = F.relu(0.05 - feature_std).mean(-1)
        # Pair frequencies estimate the equal-channel/equal-bucket sampling law.
        frequency = pair_counts / self.anchor_pair_count
        active_fraction = eligible.any(1).float().mean()
        loss = ((cohesion + 0.1 * anti_collapse) * frequency).sum() * active_fraction
        fraction = (positives / pair_counts.clamp_min(1) * frequency).sum() * active_fraction
        return torch.nan_to_num(loss, nan=0., posinf=1.), fraction.detach()

    def _sampled_anchor_loss(self, tokens, valid):
        """Approximate semantic cohesion with sampled pairs or legacy anchors."""
        patches = tokens[valid]
        if patches.shape[0] < 2:
            zero = tokens.sum() * 0
            return zero, zero.detach()

        patches = F.normalize(
            torch.nan_to_num(
                patches.float(), nan=0.0, posinf=1e4, neginf=-1e4
            ),
            p=2,
            dim=-1,
            eps=1e-6,
        )

        anchor_count = min(self.anchor_count, patches.shape[0])
        anchor_indices = torch.randperm(
            patches.shape[0], device=patches.device
        )[:anchor_count]
        similarities = (patches @ patches[anchor_indices].T).clamp(-1, 1)

        row_indices = torch.arange(patches.shape[0], device=patches.device)
        non_self = row_indices[:, None] != anchor_indices[None]
        with torch.no_grad():
            positive = similarities.detach().ge(self.anchor_threshold) & non_self

        cohesion = ((1 - similarities) * positive).sum() / positive.sum().clamp_min(1)

        # The cohesion objective alone admits identical tokens. Keep the same
        # bounded variance floor as the stable legacy auxiliary loss.
        feature_std = patches.std(dim=0, unbiased=False)
        anti_collapse = F.relu(0.05 - feature_std).mean()
        loss = cohesion + 0.1 * anti_collapse
        positive_fraction = positive.float().mean()
        return torch.nan_to_num(loss, nan=0.0, posinf=1.0), positive_fraction

    def sampled_anchor_loss(self, tokens, valid, lengths=None, channels=None):
        """Anchor loss globally or as an equal average across variables.

        Flattened patch tokens are batch-major ``[B*C,K,D]``. The channel
        variants only compare tokens from the same variable across a batch.
        ``channel_length_balanced`` additionally gives the four hard-length
        ranges 1, 2-3, 4-7, and >=8 equal weight within each variable.
        In pair mode, these weights define the sampling probabilities under a
        single batch-wide P budget; their finite-sample estimates may vary.
        """
        if self.anchor_pair_count:
            return self._pair_anchor_loss(tokens, valid, lengths, channels)
        if self.anchor_variant == 'global':
            return self._sampled_anchor_loss(tokens, valid)
        if channels is None or channels < 1 or tokens.shape[0] % channels:
            raise ValueError('Channel anchor variants require a valid channel count')
        if lengths is None or lengths.shape != valid.shape:
            raise ValueError('Channel anchor variants require patch lengths')

        channel_losses, channel_fractions = [], []
        for channel in range(channels):
            channel_tokens = tokens[channel::channels]
            channel_valid = valid[channel::channels]
            channel_lengths = lengths[channel::channels]
            if self.anchor_variant == 'channel_min4':
                channel_valid = channel_valid & channel_lengths.ge(4)
            if self.anchor_variant == 'channel_length_balanced':
                groups = (
                    channel_lengths.eq(1),
                    channel_lengths.ge(2) & channel_lengths.le(3),
                    channel_lengths.ge(4) & channel_lengths.le(7),
                    channel_lengths.ge(8),
                )
                group_results = [
                    self._sampled_anchor_loss(channel_tokens, channel_valid & group)
                    for group in groups if (channel_valid & group).sum() >= 2
                ]
                if group_results:
                    loss = torch.stack([item[0] for item in group_results]).mean()
                    fraction = torch.stack([item[1] for item in group_results]).mean()
                else:
                    loss = channel_tokens.sum() * 0
                    fraction = loss.detach()
            else:
                loss, fraction = self._sampled_anchor_loss(channel_tokens, channel_valid)
            channel_losses.append(loss)
            channel_fractions.append(fraction)
        return torch.stack(channel_losses).mean(), torch.stack(channel_fractions).mean()

    @staticmethod
    def _linear_fit_sse(samples, mask, positions):
        """Least-squares linear-fit SSE for every local candidate window."""
        weight = mask.to(samples.dtype).unsqueeze(0)
        x = positions.to(samples.dtype).view(1, 1, -1)
        count = weight.sum(-1).clamp_min(1)
        sum_x = (weight * x).sum(-1)
        sum_x2 = (weight * x.square()).sum(-1)
        sum_y = (samples * weight).sum(-1)
        sum_y2 = (samples.square() * weight).sum(-1)
        sum_xy = (samples * weight * x).sum(-1)
        centered_x2 = (sum_x2 - sum_x.square() / count).clamp_min(0)
        centered_y2 = (sum_y2 - sum_y.square() / count).clamp_min(0)
        covariance = sum_xy - sum_x * sum_y / count
        explained = torch.where(
            centered_x2 > 1e-8,
            covariance.square() / centered_x2.clamp_min(1e-8),
            torch.zeros_like(covariance),
        )
        return (centered_y2 - explained).clamp_min(0), centered_y2

    def _merge_labels(self, gains):
        """Labels on the raw grid; sparse peaks compete only at eligible gaps.

        Five candidate positions and a 0.02 background target reproduce the
        sparse-boundary experiment. Exact/near ties are retained, as there.
        Labels are constructed under no_grad by the training caller.
        """
        if self.merge_mode == 'dense':
            return torch.sigmoid((gains - self.merge_threshold) / self.merge_temperature)
        target = torch.full_like(gains, 0.02)
        eligible = gains[:, self.small_patch_len - 1::self.small_patch_len]
        if eligible.shape[1] == 0:
            return target
        local_max = F.max_pool1d(eligible.unsqueeze(1), 5, stride=1, padding=2).squeeze(1)
        peaks = eligible.ge(local_max - 1e-7) & eligible.gt(self.merge_threshold)
        target[:, self.small_patch_len - 1::self.small_patch_len] = torch.where(
            peaks,
            torch.sigmoid((eligible - self.merge_threshold) / self.merge_temperature),
            eligible.new_full((), 0.02),
        )
        return target

    def merge_supervision(self, logits, targets):
        """BCE or the validated local ranking + calibration objective.

        Radius two is measured on the eligible small-patch gap grid. Targets
        are history-only pseudo labels; no gradients flow into label generation.
        Shared-boundary mode uses the existing channel-averaged soft targets.
        """
        if not logits.numel():
            return logits.sum() * 0
        targets = targets.detach()
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        if self.merge_mode != 'sparse_rank':
            return bce
        positives = targets.gt(.5)
        neighbours = F.pad(logits, (2, 2)).unfold(1, 5, 1)
        negative = F.pad(~positives, (2, 2), value=False).unfold(1, 5, 1)
        pairs = positives[..., None] & negative
        ordering = F.softplus(1 + neighbours - logits[..., None])
        rank = (ordering * pairs).sum() / pairs.sum().clamp_min(1)
        # This matches probability mass, not the thresholded/constrained count.
        rate = (logits.sigmoid().mean(1) - positives.to(logits.dtype).mean(1)).square().mean()
        return rank + .25 * bce + rate

    @torch.no_grad()
    def waveform_lobe_targets(self, values):
        """History-only half-wave labels on the eligible small-patch gap grid.

        Linear detrending, smoothing radius merge_window in raw points,
        fixed eight-point edge exclusion, and .98/.02 soft labels. No fit-gain,
        extrema-protection, or rank losses are computed for this mode.
        """
        rows, length = values.shape
        slots = max(0, length // self.small_patch_len - 1)
        if not slots:
            return values.new_empty((rows, 0))
        # Keep label construction numerically stable under mixed precision.
        raw = values.float() if values.dtype in (torch.float16, torch.bfloat16) else values
        t = torch.linspace(-1, 1, length, device=raw.device, dtype=raw.dtype)
        centered = raw - raw.mean(-1, keepdim=True)
        slope = (centered*t).sum(-1, keepdim=True) / t.square().sum().clamp_min(1e-8)
        residual = centered - slope*t
        radius = self.merge_window
        smooth = F.avg_pool1d(
            F.pad(residual[:, None], (radius, radius), mode='replicate'),
            2 * radius + 1, stride=1,
        ).squeeze(1)
        scale = smooth.square().mean(-1, keepdim=True).sqrt()
        active = scale > (raw.std(-1, unbiased=False, keepdim=True)*1e-4 + 1e-6)
        rising = (smooth[:, :-1] <= 0) & (smooth[:, 1:] > 0)
        falling = (smooth[:, :-1] >= 0) & (smooth[:, 1:] < 0)
        index = torch.arange(length-1, device=raw.device)
        crossings = (rising | falling) & active & (index[None] >= 8) & (index[None] < length-9)
        # Coordinate of the gap after i is i+1; nearest k*small, ties to right.
        mapped = torch.div(index+1+self.small_patch_len//2, self.small_patch_len, rounding_mode='floor')-1
        valid = (mapped >= 0) & (mapped < slots)
        counts = torch.zeros((rows, slots), device=raw.device, dtype=torch.long)
        counts.scatter_add_(1, mapped.clamp(0, slots-1)[None].expand_as(crossings), (crossings & valid).long())
        return torch.where(counts > 0, raw.new_tensor(.98), raw.new_tensor(.02))

    def merge_targets(self, values):
        """Return normalized split gains and dense/sparse labels at all raw gaps."""
        _, length = values.shape
        if length < 2:
            empty = values.new_zeros(values.shape[0], 0)
            return empty, empty
        if self.merge_mode == 'waveform_lobe':
            eligible = self.waveform_lobe_targets(values)
            target = eligible.new_full(values[:, :-1].shape, .02)
            target[:, self.small_patch_len-1::self.small_patch_len] = eligible
            # No split-fit gain exists for waveform labels; keep diagnostics zero.
            return torch.zeros_like(target), target
        gaps = torch.arange(length - 1, device=values.device)
        offsets = torch.arange(
            -self.merge_window + 1, self.merge_window + 1,
            device=values.device,
        )
        indices = gaps[:, None] + offsets[None]
        valid = (indices >= 0) & (indices < length)
        samples = values[:, indices.clamp(0, length - 1)]
        left = valid & (offsets[None] <= 0)
        right = valid & (offsets[None] > 0)
        positions = offsets.to(values.dtype)
        whole_sse, total_variation = self._linear_fit_sse(
            samples, valid, positions
        )
        left_sse, _ = self._linear_fit_sse(samples, left, positions)
        right_sse, _ = self._linear_fit_sse(samples, right, positions)
        gain = (whole_sse - left_sse - right_sse).clamp_min(0)
        normalized = torch.where(
            total_variation > 1e-6,
            gain / total_variation.clamp_min(1e-6),
            torch.zeros_like(gain),
        ).clamp(0, 1)
        return normalized, self._merge_labels(normalized)

    def segment(self, probabilities):
        """Cut after t. Final short remainder is allowed; no point is discarded."""
        n, gaps = probabilities.shape
        age = torch.zeros(n, device=probabilities.device, dtype=torch.long)
        cuts = []
        for t in range(gaps):
            age = age + 1
            cut = ((age >= self.min_len) & (probabilities[:, t] > self.threshold)) | (age >= self.max_len)
            cuts.append(cut)
            age = torch.where(cut, 0, age)
        return torch.stack(cuts, 1) if cuts else probabilities.bool()

    def forward(self, x):
        # Flatten B and C once; all subsequent pooling is vectorized.
        b, raw_length, channels = x.shape
        if raw_length != self.token_seq_len * self.small_patch_len:
            raise ValueError('Input length must match configured seq_len')
        raw_values = x.transpose(1, 2).reshape(-1, raw_length)
        patches = raw_values.reshape(b * channels, self.token_seq_len, self.small_patch_len)
        length = self.token_seq_len
        values = patches.mean(-1) if self.small_patch_len > 1 else raw_values
        h = self.features(patches)
        point_features = h
        if self.patch_mode == 'fixed':
            logits = h.new_zeros((b * channels, length - 1))
        else:
            if self.boundary_network == 'conv':
                boundary_features = h
            else:
                boundary_features = (self.boundary.encode_features(h) if self.share_embedding
                                     else self.boundary.encode(patches if self.small_patch_len > 1 else values))
            # The same boundary network is shared across variables. In
            # adaptive_per_channel mode it sees each variable separately; adaptive
            # mode first averages variables to retain the old shared-boundary path.
            if self.patch_mode == 'adaptive_per_channel':
                boundary_h = boundary_features
            else:
                boundary_h = boundary_features.reshape(
                    b, channels, length, -1
                ).mean(dim=1)
            if self.boundary_network == 'conv':
                context = boundary_h.mean(1, keepdim=True).expand(-1, length-1, -1)
                logits = self.boundary(
                    torch.cat((boundary_h[:, :-1], boundary_h[:, 1:], context), -1)
                ).squeeze(-1)
            else:
                logits = self.boundary.score(boundary_h)
        merge_loss = logits.sum() * 0
        merge_gain_mean = logits.sum().detach() * 0
        boundary_target_mean = logits.sum().detach() * 0
        semantic_loss = logits.sum() * 0
        semantic_dissimilarity_mean = logits.sum().detach() * 0
        semantic_target_mean = logits.sum().detach() * 0
        if self.training and self.patch_mode != 'fixed' and logits.numel():
            if self.merge_weight:
                with torch.no_grad():
                    # Supervise only eligible gaps, using raw-observation windows.
                    merge_gain, boundary_target = self.merge_targets(raw_values)
                    merge_gain = merge_gain[:, self.small_patch_len - 1::self.small_patch_len]
                    boundary_target = boundary_target[:, self.small_patch_len - 1::self.small_patch_len]
                    if self.patch_mode == 'adaptive':
                        merge_gain = merge_gain.reshape(
                            b, channels, length - 1
                        ).mean(1)
                        boundary_target = boundary_target.reshape(
                            b, channels, length - 1
                        ).mean(1)
                merge_loss = self.merge_supervision(logits, boundary_target)
                merge_gain_mean = merge_gain.mean()
                boundary_target_mean = boundary_target.mean()
            if self.semantic_weight:
                with torch.no_grad():
                    semantic_dissimilarity = self.semantic_boundary_dissimilarity(h)
                    if self.patch_mode == 'adaptive':
                        semantic_dissimilarity = semantic_dissimilarity.reshape(
                            b, channels, length - 1
                        ).mean(1)
                    semantic_target = torch.sigmoid(
                        (semantic_dissimilarity - self.semantic_threshold)
                        / self.semantic_temperature
                    )
                semantic_loss = F.binary_cross_entropy_with_logits(
                    logits, semantic_target
                )
                semantic_dissimilarity_mean = semantic_dissimilarity.mean()
                semantic_target_mean = semantic_target.mean()
        if self.patch_mode == 'fixed':
            # A cut after every patch_len-th point gives contiguous,
            # non-overlapping patches. The final patch keeps any remainder.
            shared_probabilities = torch.zeros(
                b, length - 1, device=x.device, dtype=h.dtype
            )
            if self.patch_len < length:
                cut_positions = torch.arange(
                    self.patch_len - 1, length - 1,
                    self.patch_len, device=x.device
                )
            else:
                cut_positions = torch.empty(
                    0, dtype=torch.long, device=x.device
                )
            if cut_positions.numel():
                shared_probabilities[:, cut_positions] = 1
            shared_cuts = shared_probabilities.bool()
            probabilities = shared_probabilities[:, None, :].expand(
                b, channels, length-1
            ).reshape(b * channels, length-1)
            cuts = shared_cuts[:, None, :].expand(
                b, channels, length-1
            ).reshape(b * channels, length-1)
        elif self.patch_mode == 'adaptive':
            shared_probabilities = torch.sigmoid(logits)
            shared_cuts = self.segment(shared_probabilities.detach())
            probabilities = shared_probabilities[:, None, :].expand(
                b, channels, length-1
            ).reshape(b * channels, length-1)
            cuts = shared_cuts[:, None, :].expand(
                b, channels, length-1
            ).reshape(b * channels, length-1)
        else:
            probabilities = torch.sigmoid(logits)
            cuts = self.segment(probabilities.detach())

        # Direct straight-through path inherited from the old tokenizer. It
        # is exactly zero in the forward pass, while giving every boundary
        # probability a local gradient through the patch features. This helps
        # probabilities cross the hard threshold used by ``segment``.
        if self.training and self.patch_mode != 'fixed' and self.boundary_ste_scale:
            boundary_st = probabilities - probabilities.detach()
            boundary_st = F.pad(boundary_st, (1, 0)).unsqueeze(-1)
            h = h + self.boundary_ste_scale * boundary_st
        ids = F.pad(cuts.long().cumsum(1), (1, 0))
        counts = ids[:, -1] + 1
        # Keep a fixed token axis for the HDMixer-style temporal MLP. The
        # valid mask still records the actual variable number of patches.
        slot_length = self.patch_len if self.patch_mode == 'fixed' else self.min_len
        slots = (length + slot_length - 1) // slot_length
        index = torch.arange(slots, device=x.device)
        valid = index[None] < counts[:, None]
        hard = F.one_hot(ids, slots).to(h.dtype)
        membership = hard
        if self.training and self.patch_mode != 'fixed' and self.use_ordinal_st:
            # ST cuts retain exact hard ordinal positions in the forward pass.
            st_cuts = cuts.to(h.dtype) + probabilities - probabilities.detach()
            ordinal = F.pad(st_cuts.cumsum(1), (1, 0))
            scores = -(ordinal.unsqueeze(-1) - index).square() / self.temperature
            soft = scores.masked_fill(~valid[:, None], float('-inf')).softmax(-1)
            membership = hard + (soft - soft.detach())

        lengths = membership.sum(1).clamp_min(1)
        starts = lengths.cumsum(1) - lengths
        time = torch.arange(length, device=x.device, dtype=h.dtype)
        relative = (time[None, :, None] - starts[:, None] + 0.5) / lengths[:, None]
        need_reconstruction = self.training and bool(self.reconstruction_weight)
        if self.patch_pooling == 'bins' or need_reconstruction:
            bin_id = (relative.detach()*self.bins).long().clamp(0, self.bins-1)
            bin_weights = F.one_hot(bin_id, self.bins).to(h.dtype)
            if self.training and self.use_ordinal_st and self.patch_mode != 'fixed':
                centers = (torch.arange(self.bins, device=x.device, dtype=h.dtype)+0.5)/self.bins
                soft_bins = (-(relative.unsqueeze(-1)-centers).square()*self.bins**2).softmax(-1)
                bin_weights = bin_weights + (soft_bins-soft_bins.detach())
            weights = membership.unsqueeze(-1)*bin_weights
        if self.patch_pooling == 'boundary_query':
            pooled, _ = self.boundary_query_pool(h, membership, relative)
            pooled_features = pooled
        else:
            mass = weights.sum(1)
            pooled = torch.einsum('ntkr,ntd->nkrd', weights, h)/mass.clamp_min(1).unsqueeze(-1)
            # Short segments may have empty bins. Fill with their segment mean.
            mean_h = torch.einsum('ntk,ntd->nkd', membership, h)/lengths.unsqueeze(-1)
            pooled = torch.where((mass > 0).unsqueeze(-1), pooled, mean_h.unsqueeze(2))
            pooled_features = pooled.flatten(2)
        if self.patch_use_statistics:
            mean = torch.einsum('ntk,nt->nk', membership, values)/lengths
            point_second = patches.square().mean(-1) if self.small_patch_len > 1 else values.square()
            second = torch.einsum('ntk,nt->nk', membership, point_second)/lengths
            std = (second-mean.square()).clamp_min(0).add(1e-6).sqrt()
            start_idx = starts.detach().long().clamp(0, length-1)
            end_idx = (starts+lengths-1).detach().long().clamp(0, length-1)
            delta = (raw_values.gather(1, (end_idx + 1) * self.small_patch_len - 1)
                     - raw_values.gather(1, start_idx * self.small_patch_len))
            pooled_features = torch.cat((pooled_features, torch.stack((mean, std, delta), -1)), -1)
        content_token = self.project(pooled_features)
        content_token = content_token * valid.unsqueeze(-1)
        hard_lengths = hard.sum(1).detach() * self.small_patch_len
        anchor_loss = content_token.sum() * 0
        anchor_positive_fraction = anchor_loss.detach()
        if self.training and self.anchor_weight:
            anchor_loss, anchor_positive_fraction = self.sampled_anchor_loss(
                content_token, valid, hard_lengths, channels
            )
        token = content_token
        if not self.broadcast_backbone or need_reconstruction:
            geometry = torch.stack(((starts+lengths/2)/length, lengths/length), -1)
            token = (content_token+self.position(geometry))*valid.unsqueeze(-1)
        reconstruction = x.new_zeros(())
        if need_reconstruction:
            if self.small_patch_len == 1:
                restored = torch.einsum('ntkr,nkr->nt', weights, self.reconstruct(token))
            else:
                restored = torch.einsum('ntkr,nkrs->nts', weights,
                    self.reconstruct(token).reshape(b * channels, slots, self.bins, self.small_patch_len)
                ).flatten(1)
            reconstruction = F.mse_loss(restored, raw_values)
        # Raw boundary-probability penalty, NOT the expected constrained count:
        # forced max-length cuts and minimum-length suppression change that count.
        if self.training and self.count_weight and self.patch_mode != 'fixed':
            cost = (1+probabilities.sum(1)).mean()/raw_length
        else:
            cost = x.new_zeros(())
        raw_cuts = cuts
        if self.small_patch_len > 1:
            raw_cuts = cuts.new_zeros((b * channels, raw_length - 1))
            raw_cuts[:, self.small_patch_len - 1::self.small_patch_len] = cuts
        metadata = {'valid': valid, 'counts': counts.detach(), 'lengths': hard_lengths,
                    'patch_ids': ids, 'point_features': point_features,
                    # Pre-geometry patch representation: output of self.project.
                    'content_tokens': content_token,
                    'boundaries': raw_cuts.detach(), 'reconstruction': reconstruction, 'cost': cost,
                    'anchor_loss': anchor_loss,
                    'anchor_positive_fraction': anchor_positive_fraction.detach(),
                    'merge_loss': merge_loss, 'merge_gain_mean': merge_gain_mean,
                    'boundary_target_mean': boundary_target_mean,
                    'semantic_loss': semantic_loss,
                    'semantic_dissimilarity_mean': semantic_dissimilarity_mean,
                    'semantic_target_mean': semantic_target_mean,
                    'boundary_probability_mean': (
                        probabilities.detach().mean() if probabilities.numel()
                        else probabilities.detach().sum()
                    ),
                    'centers': (starts + lengths / 2) * self.small_patch_len - .5}
        return token, metadata


class HDMixerBlock(nn.Module):
    """Lightweight HDMixer block over [B, C, K, D] patch tokens."""

    def __init__(self, d, hidden, n_vars, n_patches, dropout):
        super().__init__()
        self.patch_mixer = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d),
            nn.Dropout(dropout),
        )
        self.time_mixer = nn.Sequential(
            nn.LayerNorm(n_patches),
            nn.Linear(n_patches, 2 * n_patches),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * n_patches, n_patches),
            nn.Dropout(dropout),
        )
        self.variable_mixer = nn.Sequential(
            nn.LayerNorm(n_vars),
            nn.Linear(n_vars, 2 * n_vars),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * n_vars, n_vars),
            nn.Dropout(dropout),
        )

    def forward(self, x, valid):
        # x: [B, C, K, D], valid: [B, C, K]
        mask = valid.unsqueeze(-1).to(x.dtype)
        x = x * mask

        # Feature/channel mixing, independently for every variable and patch.
        u = (x + self.patch_mixer(x)) * mask

        # Temporal mixing across the fixed patch axis, independently for each
        # variable and feature channel.
        v = u.transpose(2, 3)
        v = (v + self.time_mixer(v)).transpose(2, 3)
        v = v * mask

        # Variable mixing across all input channels, independently for every
        # patch and feature channel.
        w = v.transpose(1, 3)
        w = (w + self.variable_mixer(w)).transpose(1, 3)
        return w * mask


class PointTimeMLPBlock(nn.Module):
    """Time MLP, optional variable MLP, and FFN on dense points [B*C, T, D]."""

    def __init__(self, seq_len, d_model, inter_dim, dropout,
                 n_vars=1, channel_mlp=False, post_norm=False,
                 channel_hidden_c=False, residual_to_input=False, channel_norm_c=False,
                 time_hidden_ratio=0.0, channel_scale_init=0.1,
                 time_norm_t=False, channel_scale_learnable=True,
                 channel_hidden_ratio=2.0):
        super().__init__()
        self.n_vars = n_vars
        self.post_norm = post_norm
        self.use_channel_mlp = channel_mlp
        self.residual_to_input = residual_to_input
        self.channel_norm_c = channel_norm_c
        self.time_norm_t = bool(time_norm_t)
        channel_hidden_ratio = float(channel_hidden_ratio)
        if not math.isfinite(channel_hidden_ratio) or channel_hidden_ratio < 0:
            raise ValueError('broadcast_channel_hidden_ratio must be finite and non-negative')
        time_hidden_ratio = float(time_hidden_ratio)
        if not math.isfinite(time_hidden_ratio) or time_hidden_ratio < 0:
            raise ValueError('broadcast_time_hidden_ratio must be finite and non-negative')
        # N is the dense history length here, before prediction-head pooling.
        time_hidden = (inter_dim if time_hidden_ratio == 0
                       else max(1, int(seq_len * time_hidden_ratio)))
        self.time_norm = nn.LayerNorm(seq_len if self.time_norm_t else d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(seq_len, time_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(time_hidden, seq_len), nn.Dropout(dropout),
        )
        self.feature_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, inter_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(inter_dim, d_model), nn.Dropout(dropout),
        )
        if channel_mlp:
            channel_scale_init = float(channel_scale_init)
            if not math.isfinite(channel_scale_init):
                raise ValueError('broadcast_channel_scale_init must be finite')
            channel_hidden = (
                max(1, int(n_vars * channel_hidden_ratio))
                if channel_hidden_ratio > 0
                else (n_vars if channel_hidden_c else 2 * n_vars)
            )
            self.channel_norm = nn.LayerNorm(n_vars if channel_norm_c else d_model)
            self.channel_mlp = nn.Sequential(
                nn.Linear(n_vars, channel_hidden), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(channel_hidden, n_vars), nn.Dropout(dropout),
            )
            # One residual scale per output variable. A small initialization
            # lets the model introduce cross-variable information gradually.
            channel_scale = torch.full((n_vars,), channel_scale_init)
            if channel_scale_learnable:
                self.channel_scale = nn.Parameter(channel_scale)
            else:
                self.register_buffer('channel_scale', channel_scale)

    def normalize_time(self, x):
        if not self.time_norm_t:
            return self.time_norm(x)
        return self.time_norm(x.transpose(1, 2)).transpose(1, 2)

    def normalize_channels(self, x):
        if not self.channel_norm_c:
            return self.channel_norm(x)
        # Normalize C separately for each sample, time point and feature.
        n, t, d = x.shape
        x = x.reshape(-1, self.n_vars, t, d).permute(0, 2, 3, 1)
        return self.channel_norm(x).permute(0, 3, 1, 2).reshape(n, t, d)

    def mix_channels(self, x):
        n, t, d = x.shape
        x = x.reshape(-1, self.n_vars, t, d).permute(0, 2, 3, 1)
        return self.channel_mlp(x).permute(0, 3, 1, 2).reshape(n, t, d)

    def scale_channel_update(self, x):
        n, t, d = x.shape
        x = x.reshape(-1, self.n_vars, t, d)
        return (x * self.channel_scale[None, :, None, None]).reshape(n, t, d)

    def forward(self, x):
        # Each block owns its reference; sublayers still process sequentially.
        residual = x
        if self.post_norm:
            x = self.normalize_time(
                x + self.time_mlp(x.transpose(1, 2)).transpose(1, 2)
            )
            if self.use_channel_mlp:
                base = residual if self.residual_to_input else x
                update = self.scale_channel_update(self.mix_channels(x))
                x = self.normalize_channels(base + update)
            base = residual if self.residual_to_input else x
            return self.feature_norm(base + self.ffn(x))
        x = x + self.time_mlp(self.normalize_time(x).transpose(1, 2)).transpose(1, 2)
        if self.use_channel_mlp:
            base = residual if self.residual_to_input else x
            update = self.scale_channel_update(
                self.mix_channels(self.normalize_channels(x))
            )
            x = base + update
        base = residual if self.residual_to_input else x
        return base + self.ffn(self.feature_norm(x))


class CompactATM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.normalize = Normalize(config.enc_in, affine=False)
        self.time_tokenizer = CompactTokenizer(config)
        self.encoder_type = config.encoder_type
        self.prediction_backbone = getattr(config, 'prediction_backbone', 'token_encoder').lower()
        self.last_patch_stats = None
        if self.prediction_backbone == 'patch_broadcast_mlp':
            # Broadcast pre-position self.project tokens to points in their hard
            # patches, then forecast with a dense Time MLP + FFN trunk.
            self.broadcast_projection = (
                nn.Linear(config.d_model, config.d_model)
                if self.time_tokenizer.patch_use_statistics else nn.Identity()
            )
            self.point_encoder = nn.ModuleList([
                PointTimeMLPBlock(
                    self.time_tokenizer.token_seq_len, config.d_model, config.inter_dim, config.dropout,
                    n_vars=config.enc_in,
                    channel_mlp=bool(getattr(config, 'broadcast_channel_mlp', False)),
                    post_norm=bool(getattr(config, 'mlp_post_norm', False)),
                    channel_hidden_c=bool(getattr(config, 'broadcast_channel_hidden_c', False)),
                    channel_hidden_ratio=getattr(config, 'broadcast_channel_hidden_ratio', 2.0),
                    residual_to_input=bool(getattr(config, 'broadcast_residual_to_input', False)),
                    channel_norm_c=bool(getattr(config, 'broadcast_channel_norm_c', False)),
                    time_hidden_ratio=getattr(config, 'broadcast_time_hidden_ratio', 0.0),
                    channel_scale_init=getattr(config, 'broadcast_channel_scale_init', 0.1),
                    time_norm_t=bool(getattr(config, 'broadcast_time_norm_t', False)),
                    channel_scale_learnable=bool(
                        getattr(config, 'broadcast_channel_scale_learnable', True)
                    ),
                )
                for _ in range(config.encoder_layers)
            ])
            self.prediction_feature_ratio = float(getattr(config, 'prediction_feature_ratio', 0.0))
            if not 0 <= self.prediction_feature_ratio <= 1:
                raise ValueError('prediction_feature_ratio must be between 0 and 1')
            # Zero retains the legacy scalar readout; one keeps features intact.
            self.prediction_feature_dim = max(1, int(config.d_model * self.prediction_feature_ratio))
            self.point_readout = (
                nn.Identity() if self.prediction_feature_ratio == 1
                else nn.Linear(config.d_model, self.prediction_feature_dim)
            )
            # Zero preserves the original full-length head. A positive value
            # adaptively average-pools the dense time axis before this head.
            self.prediction_pool_len = int(getattr(config, 'prediction_pool_len', 0))
            if self.prediction_pool_len < 0:
                raise ValueError('prediction_pool_len must be zero or positive')
            head_input_len = self.prediction_pool_len or self.time_tokenizer.token_seq_len
            self.point_head = nn.Sequential(
                nn.Dropout(config.head_dropout),
                nn.Linear(head_input_len * self.prediction_feature_dim, config.pred_len)
            )
            return
        if self.prediction_backbone != 'token_encoder':
            raise ValueError('prediction_backbone must be token_encoder or patch_broadcast_mlp')
        if self.encoder_type == 'forecast_token_attn':
            self.encoder = ForecastTokenEncoder(config)
            return
        if self.encoder_type != 'hdmixer':
            raise ValueError('encoder_type must be forecast_token_attn or hdmixer')
        patch_mode = getattr(config, 'patch_mode', 'fixed').lower()
        slot_length = config.patch_len if patch_mode == 'fixed' else config.patch_min_len
        max_patches = (config.seq_len + slot_length - 1) // slot_length
        self.encoder = nn.ModuleList([
            HDMixerBlock(
                config.d_model,
                config.inter_dim,
                config.enc_in,
                max_patches,
                config.dropout,
            )
            for _ in range(config.encoder_layers)
        ])
        self.queries = nn.Parameter(torch.randn(config.patch_readout_queries, config.d_model)*0.02)
        self.head = nn.Sequential(nn.Dropout(config.head_dropout),
                                 nn.Linear(config.patch_readout_queries*config.d_model, config.pred_len))
        self.last_patch_stats = None

    def forward(self, x_enc, *args, **kwargs):
        b, _, c = x_enc.shape
        x = self.normalize(x_enc, 'norm')
        tokens, meta = self.time_tokenizer(x)
        valid = meta['valid']
        stat_keys = (
            'counts', 'lengths', 'boundaries', 'merge_gain_mean',
            'boundary_target_mean', 'boundary_probability_mean',
            'anchor_loss', 'anchor_positive_fraction',
            'semantic_loss', 'semantic_dissimilarity_mean',
            'semantic_target_mean',
        )
        self.last_patch_stats = {key: meta[key].detach() for key in stat_keys}
        aux = tokens.sum()*0
        if self.training:
            aux = (
                self.config.patch_reconstruction_weight * meta['reconstruction']
                + self.config.patch_count_weight * meta['cost']
                + self.config.patch_anchor_weight * meta['anchor_loss']
                + self.config.boundary_merge_weight * meta['merge_loss']
                + self.config.boundary_semantic_weight * meta['semantic_loss']
            )
        if self.prediction_backbone == 'patch_broadcast_mlp':
            patch_ids = meta['patch_ids']
            patch_features = self.broadcast_projection(meta['content_tokens'])
            broadcast = patch_features.gather(
                1, patch_ids.unsqueeze(-1).expand(-1, -1, patch_features.shape[-1])
            )
            points = meta['point_features'] + broadcast
            for block in self.point_encoder:
                points = block(points)
            # [B*C, T, R] -> [B*C, R, T]; pool only over time.
            point_values = self.point_readout(points).transpose(1, 2)
            if self.prediction_pool_len:
                point_values = F.adaptive_avg_pool1d(
                    point_values, self.prediction_pool_len
                )
            prediction = self.point_head(point_values.flatten(1))
            prediction = prediction.reshape(b, c, -1).transpose(1, 2)
            return self.normalize(prediction, 'denorm'), aux
        tokens = tokens.reshape(b, c, tokens.shape[1], tokens.shape[2])
        valid = valid.reshape(b, c, valid.shape[1])
        if self.encoder_type == 'forecast_token_attn':
            prediction = self.encoder(tokens, valid, meta['centers'].reshape(b, c, -1), x_enc.shape[1])
            return self.normalize(prediction, 'denorm'), aux
        for block in self.encoder:
            tokens = block(tokens, valid)
        tokens = tokens.reshape(b * c, tokens.shape[2], tokens.shape[3])
        valid = valid.reshape(b * c, valid.shape[2])
        scores = torch.einsum('qd,nkd->nqk', self.queries, tokens)/(tokens.shape[-1]**0.5)
        attention = scores.masked_fill(~valid[:, None], float('-inf')).softmax(-1)
        summary = torch.einsum('nqk,nkd->nqd', attention, tokens)
        prediction = self.head(summary.flatten(1)).reshape(b, c, -1).transpose(1, 2)
        return self.normalize(prediction, 'denorm'), aux
