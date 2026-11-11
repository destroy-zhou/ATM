import math
import numpy as np
import torch
import torch.nn as nn
import enum
from ts_benchmark.baselines.mmpd.models.loss_funcs.mmpd.gaussian_mixture import DiffusionVariationalGaussianMixture, multi_mode_summary

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    """
    This is the deprecated API for creating beta schedules.
    See get_named_beta_schedule() for the new library of schedules.
    """
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.
    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        return get_beta_schedule(
            "linear",
            beta_start=scale * 0.0001,
            beta_end=scale * 0.02,
            num_diffusion_timesteps=num_diffusion_timesteps,
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.9):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def extract(nums_seq, t, broadcast_shape):
    #get the number from nums_seq according to t
    #nums_seq: [timesteps]
    #t: [*]

    out = nums_seq[t] #same shape as t

    while len(out.shape) < len(broadcast_shape):
        out = out[..., None]

    return out + torch.zeros(broadcast_shape, device=t.device)

class GaussianDiffusion(nn.Module):
    """
    Utilities for training and sampling diffusion models.
    Original ported from this codebase:
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42
    :param betas: a 1-D numpy array of betas for each diffusion timestep,
                  starting at T and going to 1.
    """

    def __init__(
        self,
        *,
        betas,
    ):
        super(GaussianDiffusion, self).__init__()

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        betas = torch.from_numpy(np.array(betas))
        register_buffer("betas", betas)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        register_buffer("alphas", alphas)
        register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        register_buffer("alphas_cumprod_prev", torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]]))
        register_buffer("alphas_cumprod_next", torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0])]))
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - self.alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        register_buffer("posterior_variance", self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        register_buffer("posterior_log_variance_clipped", torch.log(torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])))

        register_buffer("posterior_mean_coef1", self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        register_buffer("posterior_mean_coef2", (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

        #find the t where alphas_cumprod is closest to 0.5
        self.anchor_t = torch.argmin(torch.abs(self.alphas_cumprod - 0.5))


    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: clean patches; in [batch_size, patch_num, patch_size]
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step; in [batch_size]
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.
        In other words, sample from q(x_t | x_0).
        :param x_start: the initial data batch; in [batch_size, patch_num, patch_size]
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step; in [batch_size]
        :param noise: if specified, the iid normal noise; in [batch_size, patch_num, patch_size]
        :return: A noisy version of x_start; in [batch_size, patch_num, patch_size]
        """
        if noise is None:
            noise = torch.randn(x_start.shape).to(x_start.device)
        assert noise.shape == x_start.shape
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, model, x, condition, t):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the noisy patch at time t; in [batch_size, patch_num, patch_size]
        :param condition: the condition for the model; in [batch_size, patch_num, d_condition]
        :param t: a 1-D Tensor of timesteps.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """

        batch_size, patch_num, patch_size = x.shape
        assert t.shape == (batch_size,)
        model_output = model(x, condition, t)

        model_variance, model_log_variance = self.posterior_variance, self.posterior_log_variance_clipped
        model_variance = extract(model_variance, t, x.shape)
        model_log_variance = extract(model_log_variance, t, x.shape)

        pred_xstart = self._predict_xstart_from_eps(x_t=x, t=t, eps=model_output)

        model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)

        assert model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape
        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
        }

    def _predict_xstart_from_eps(self, x_t, t, eps):   #rearrange Eq. 4 of DDPM
        assert x_t.shape == eps.shape
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def training_losses(self, model, x_start, condition, t, noise=None, point_weight=0.0):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the clean patch; in [batch_size, patch_num, patch_size]
        :param condition: the condition for the model; in [batch_size, patch_num, d_condition]
        :param t: a batch of timestep indices.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """

        if noise is None:
            noise = torch.randn(x_start.shape).to(x_start.device)
        x_t = self.q_sample(x_start, t, noise=noise)

        terms = {}


        model_output = model(x_t, condition, t)

        target = noise
        assert model_output.shape == target.shape == x_start.shape
        diffusion_mse_nonreduced = (model_output - target) ** 2
        terms["diffusion_mse"] = diffusion_mse_nonreduced.sum(dim=(-1, -2))

        # point prediction loss
        x_t_zeros = torch.zeros(x_start.shape).to(x_start.device)
        anchor_t = torch.tensor([self.anchor_t] * x_start.shape[0]).to(x_start.device)
        anchor_noise = self._predict_eps_from_xstart(x_t=x_t_zeros, t=anchor_t, pred_xstart=x_start)
        anchor_output = model(x_t_zeros, condition, anchor_t)

        point_loss_nonreduced = (anchor_noise - anchor_output) ** 2
        terms["point_loss"] = point_loss_nonreduced.sum(dim=(-1, -2))
        terms["loss"] = (1 - point_weight) * terms["diffusion_mse"] + point_weight * terms["point_loss"]

        return terms

    @torch.no_grad()
    def point_pred(
        self,
        model,
        patches_shape,
        condition,
    ):

        batch_size, patch_num, patch_size = patches_shape
        zeros = torch.zeros(patches_shape).to(condition.device)
        anchor_t = torch.tensor([self.anchor_t] * batch_size).to(zeros.device)
        assert anchor_t.shape == (batch_size,)
        model_output = model(zeros, condition, anchor_t)

        pred_xstart = self._predict_xstart_from_eps(x_t=zeros, t=anchor_t, eps=model_output)

        return pred_xstart

    @torch.no_grad()
    def p_sample(
        self,
        model,
        x,
        condition,
        t,
        temperature=1.0
    ):
        """
        Sample x_{t-1} from the model at the given timestep.
        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param condition: the condition for the model; in [batch_size, patch_num, d_condition]
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param temperature: temperature scaling during Diff Loss sampling.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        out = self.p_mean_variance(
            model,
            x,
            condition,
            t,
        )

        noise = torch.randn(x.shape).to(x.device)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )  # no noise when t == 0
        # scale the noise by temperature
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise * temperature
        return {"sample": sample, "pred_xstart": out["pred_xstart"]}

    @torch.no_grad()
    def p_sample_loop(
        self,
        model,
        x_shape,
        condition,
        sample_num=1,
        temperature=1.0,
        gmm=True,
        gmm_components=10,
        prior_pi_decay=0.5,
        prior_precision_shape=1e3,
        gmm_iterations=10
    ):
        """
        Generate samples from the model and yield intermediate samples from
        each timestep of diffusion.
        Arguments are the same as p_sample_loop().
        Returns a generator over dicts, where each dict is the return value of
        p_sample().
        """
        batch_size, patch_num, patch_size = x_shape
        _, _, d_condition = condition.shape
        folded_xt  = torch.randn([batch_size*sample_num, patch_num, patch_size]).to(condition.device)
        folded_condition = condition[:, None, :, :].expand(-1, sample_num, -1, -1).reshape(batch_size*sample_num, patch_num, d_condition)
        indices = list(range(self.num_timesteps))[::-1]

        xt = folded_xt.reshape(batch_size, sample_num, patch_num * patch_size)

        if gmm:
            gmm = DiffusionVariationalGaussianMixture(
                n_components = gmm_components,
                alphas_cumprod = self.alphas_cumprod,
                prior_pi_decay = prior_pi_decay,
                prior_precision_shape = prior_precision_shape,
                batch_x = xt
            )

        for i in indices:
            if gmm:
                xt = folded_xt.reshape(batch_size, sample_num, patch_num * patch_size)
                for _ in range(gmm_iterations):
                    log_resp = gmm.e_step(xt)
                    gmm.m_step(xt, log_resp, i)

            t = torch.tensor([i] * (batch_size*sample_num)).to(condition.device)
            out = self.p_sample(
                model,
                folded_xt,
                folded_condition,
                t,
                temperature=temperature,
            )
            folded_xt = out["sample"]

        if gmm:
            xt = folded_xt.reshape(batch_size, sample_num, patch_num * patch_size)
            for _ in range(gmm_iterations):
                log_resp = gmm.e_step(xt)
                gmm.m_step(xt, log_resp, i)


            log_resp = gmm.predict(xt) # [batch_size, sample_num, n_components]
            assigned_cluster = torch.argmax(log_resp, dim=-1) # [batch_size, sample_num]

            xt = folded_xt.reshape(batch_size, sample_num, patch_num * patch_size)

            num_in_mode, mode_median, confidence_interval = multi_mode_summary(xt, assigned_cluster, gmm_components, confidence=[0.5, 0.9])

            gmm_results = {
                "assigned_cluster": assigned_cluster,
                "num_in_mode": num_in_mode,
                "mode_median": mode_median,
                "confidences": [0.5, 0.9],
                "confidence_intervals": confidence_interval,
            }

        else:
            gmm_results = None

        denoised_x0 = out["sample"].view(batch_size, sample_num, patch_num, patch_size)

        return denoised_x0, gmm_results
