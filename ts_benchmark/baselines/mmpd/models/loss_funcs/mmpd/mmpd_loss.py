import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F
from einops import rearrange
from math import ceil

from ts_benchmark.baselines.mmpd.models.loss_funcs.mmpd.patch_consistent_mlp import Patch_Consistent_MLP
from ts_benchmark.baselines.mmpd.models.loss_funcs.mmpd.diffusion_respace import create_diffusion
from ts_benchmark.baselines.mmpd.models.loss_funcs.base_loss import BaseLossFunc

class MMPD_Loss(BaseLossFunc):
    def __init__(self, configs):
        super(MMPD_Loss, self).__init__()

        self.d_condition = configs.d_model
        self.d_diffusion = configs.d_diffusion
        self.patch_size = configs.patch_size
        self.out_len = configs.out_len
        self.diffusion_layers = configs.diffusion_layers
        self.max_diffusion_steps = configs.max_diffusion_steps
        self.beta_schedule = configs.beta_schedule
        self.radius = configs.radius
        self.num_sampling_steps = configs.num_sampling_steps
        self.dropout = configs.dropout

        self.net = Patch_Consistent_MLP(self.max_diffusion_steps, self.d_condition, self.d_diffusion, self.patch_size, self.diffusion_layers, \
                                     self.radius, dropout=0.0, data_dim=configs.data_dim)

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule=self.beta_schedule, diffusion_steps=self.max_diffusion_steps)
        self.gen_diffusion = create_diffusion(timestep_respacing=self.num_sampling_steps, noise_schedule=self.beta_schedule, diffusion_steps=self.max_diffusion_steps)

    def compute_loss(self, target_seq, dec_condition, point_weight=0.0, *args, **kwargs):
        #target_seq: [batch_size, seq_len]
        #dec_condition: [batch_size, patch_num, d_condition]

        batch_size, data_dim, seq_len = target_seq.shape
        _, _, patch_num, _ = dec_condition.shape

        assert patch_num == ceil(self.out_len / self.patch_size), f"Each token corresponds to a patch of length {self.patch_size}, so the number of patches should be ceil(out_len / patch_size)={ceil(self.out_len / self.patch_size)}, but got {patch_num}."

        target_seq = rearrange(target_seq, 'b d l -> (b d) l')
        dec_condition = rearrange(dec_condition, 'b d n p -> (b d) n p')

        flatten_batch_size = target_seq.shape[0]

        #pad the target_seq if not multiple of patch_size
        point_to_pad = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        if point_to_pad > 0:
            target_seq = torch.cat([target_seq, target_seq[:, -1:].expand(-1, point_to_pad)], dim=1)

        target_patches = rearrange(target_seq, 'b (n p) -> b n p', p=self.patch_size)
        t = torch.randint(0, self.max_diffusion_steps, (flatten_batch_size,)).to(dec_condition.device).long()

        loss_dict = self.train_diffusion.training_losses(self.net, target_patches, dec_condition, t, point_weight=point_weight)
        loss = loss_dict["loss"]

        loss = rearrange(loss, '(b d) -> b d', d=data_dim)

        return loss

    @torch.no_grad()
    def predict(self, dec_condition, prob_pred = False,
        sample_num = 1, temperature = 1.0, \
        gmm_components=5, prior_pi_decay=0.5, prior_precision_shape=1e3, gmm_iterations=10, *args, **kwargs):

        batch_size, data_dim, patch_num, _ = dec_condition.shape
        dec_condition = rearrange(dec_condition, 'b d n p -> (b d) n p')
        flatten_batch_size = dec_condition.shape[0]

        deterministic_patches = self.train_diffusion.point_pred(
            model = self.net,
            patches_shape = [flatten_batch_size, patch_num, self.patch_size],
            condition = dec_condition
        )
        deterministic_seq = rearrange(deterministic_patches, 'b n p -> b (n p)')[:, :self.out_len] # truncate to out_len
        deterministic_pred = rearrange(deterministic_seq, '(batch_size data_dim) seq_len -> batch_size data_dim seq_len', data_dim=data_dim)
        multi_mode_pred, prob_samples = None, None

        if prob_pred:
            sampled_patches, gmm_results = self.gen_diffusion.p_sample_loop(
                model=self.net,
                x_shape=[flatten_batch_size, patch_num, self.patch_size],
                condition=dec_condition,
                sample_num=sample_num,
                temperature=temperature,
                gmm_components=gmm_components,
                prior_pi_decay=prior_pi_decay,
                prior_precision_shape=prior_precision_shape,
                gmm_iterations=gmm_iterations
            )

            sampled_seqs = rearrange(sampled_patches, 'b s n p -> b s (n p)')[:, :, :self.out_len] # truncate to out_len
            prob_samples = rearrange(sampled_seqs, '(batch_size data_dim) sample_num seq_len -> batch_size data_dim sample_num seq_len', data_dim=data_dim)

            num_in_mode = rearrange(gmm_results['num_in_mode'], '(batch_size data_dim) mode_num -> batch_size data_dim mode_num', data_dim=data_dim)
            mode_center = rearrange(gmm_results['mode_median'], '(batch_size data_dim) mode_num seq_len -> batch_size data_dim mode_num seq_len', data_dim=data_dim)[..., :self.out_len]
            confidences = gmm_results['confidences']
            mode_CI = rearrange(gmm_results['confidence_intervals'], '(batch_size data_dim) mode_num conf_num c seq_len -> batch_size data_dim mode_num conf_num c seq_len', data_dim=data_dim)[..., :self.out_len]
            mode_prob = 1.0 * num_in_mode / num_in_mode.sum(dim=-1, keepdim=True)

            multi_mode_pred = {
                'mode_prob': mode_prob,
                'mode_center': mode_center,
                'confidences': confidences,
                'mode_CI': mode_CI
            }

        return deterministic_pred, multi_mode_pred, prob_samples
