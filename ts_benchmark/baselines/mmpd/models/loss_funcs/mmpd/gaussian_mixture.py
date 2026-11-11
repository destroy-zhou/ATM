import torch
import math
from copy import deepcopy

#varitional inference for p(pi) and p(gamma), deterministic inference for mu
class DiffusionVariationalGaussianMixture:
    def __init__(
        self,
        n_components = 1,
        alphas_cumprod = None,
        prior_pi_decay = 0.2,
        prior_precision_shape = 2.0,
        batch_x = None
    ):

        self.n_components = n_components
        self.alphas_cumprod = alphas_cumprod
        self.timestamps = len(alphas_cumprod)

        batch_size, n_samples, n_features = batch_x.shape
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.n_features = n_features
        self.device = batch_x.device

        ##############  set prior distribution for parameters of GMM at each timestamp (corresponding to diffusion process)  ##############
        #prior for pi: Dirichlet distribution, remains constant for all timestamps
        self.prior_pi_concentration = torch.tensor([1.0 * prior_pi_decay**i for i in range(n_components)]).to(self.device)

        #prior for precision: Gamma distribution, shape remains constant, rate changes to align the mode to diffusion variance
        self.prior_precision_shape = prior_precision_shape
        self.prior_precision_rate = ((prior_precision_shape) * (1 - self.alphas_cumprod)).to(self.device)

        ##############  Init posterior distribution for parameters of GMM at T  ##############
        self.post_pi_concentration = deepcopy(self.prior_pi_concentration)[None, :].expand(batch_size, -1)

        self.post_precision_shape = torch.ones(batch_size, n_components).to(self.device) * self.prior_precision_shape
        self.post_precision_rate = torch.ones(batch_size, n_components).to(self.device) * self.prior_precision_rate[-1]

        self.post_mu = torch.zeros(batch_size, n_components, n_features).to(self.device)


    def e_step(self, batch_x):
        # batch_x: [batch_size, n_samples, n_features]
        batch_size, n_samples, n_features = batch_x.shape

        #gaussian prob term
        expected_precision = self.post_precision_shape / self.post_precision_rate # [batch_size, n_components]
        log_prob_term1 = expected_precision[:, None, :] * ((batch_x[:, :, None, :] - self.post_mu[:, None, :, :])**2).sum(-1)
        log_prob_term2 = n_features * math.log(2 * math.pi)
        log_prob_term3 = n_features * (torch.digamma(self.post_precision_shape) - torch.log(self.post_precision_rate))
        log_gaussian_prob = -0.5 * (log_prob_term1 + log_prob_term2 - log_prob_term3[:, None, :]) # [batch_size, n_samples, n_components]

        #weight term
        log_pi = torch.digamma(self.post_pi_concentration) - torch.digamma(self.post_pi_concentration.sum(-1, keepdim=True)) # [batch_size, n_components]

        weighted_log_prob = log_gaussian_prob + log_pi[:, None, :]
        log_prob_norm = torch.logsumexp(weighted_log_prob, dim=-1, keepdim = True) # [batch_size, n_samples]
        log_resp = weighted_log_prob - log_prob_norm

        return log_resp

    def m_step(self, batch_x, log_resp, t):
        # batch_x: [batch_size, n_samples, n_features]
        # log_resp: [batch_size, n_samples, n_components]
        # t: current timestamp

        n_features = batch_x.shape[-1]
        resp = torch.exp(log_resp) # [batch_size, n_samples, n_components]

        #update parameters related to q(pi)
        nk = resp.sum(dim=1) # [batch_size, n_components]
        new_post_pi_concentration = self.prior_pi_concentration[None, :] + nk

        #compute optimal mu_t
        new_post_mu = (resp[:, :, :, None] * batch_x[:, :, None, :]).sum(1) / (nk[:, :, None] + 10 * torch.finfo(resp.dtype).eps) # [batch_size, n_components, n_features]

        #update parameters related to q(gamma)
        new_post_precision_shape = self.prior_precision_shape + 0.5 * n_features * nk
        weighted_var = (resp * ((batch_x[:, :, None, :] - new_post_mu[:, None, :, :])**2).sum(-1)).sum(1) # [batch_size, n_components]
        new_post_precision_rate = self.prior_precision_rate[t] + 0.5 * weighted_var

        self.post_pi_concentration = new_post_pi_concentration
        self.post_mu = new_post_mu
        self.post_precision_shape = new_post_precision_shape
        self.post_precision_rate = new_post_precision_rate

        return

    def predict(self, batch_x):
        # batch_x: [batch_size, n_samples, n_features]
        log_resp = self.e_step(batch_x)
        return log_resp

def multi_mode_summary(samples, assigned_cluster, num_modes = 10, confidence=[0.5, 0.9]):
    '''
    We assume there are multiple modes in the final samples, and each sample have been assigned to one mode by assigned_cluster
    Now we get the median and several confidence interval of each mode and rank them by the probability
    samples: [batch_size, sample_num, n_features]
    assigned_cluster: [batch_size, sample_num]
    '''

    batch_size, sample_num, n_features = samples.shape

    batch_num_in_mode = []
    batch_mode_median = []
    batch_confidence_interval = []

    for i in range(batch_size):
        num_in_mode = []
        mode_median = []
        mode_interval = []
        for j in range(num_modes):
            assignment = (assigned_cluster[i] == j)
            if (assignment.sum() > 0):
                samples_in_cluster = samples[i][assignment]
                median_curve = torch.quantile(samples_in_cluster, 0.5, dim=0) #[n_features]

                confidence_intervals = []
                for c in confidence:
                    down_curve = torch.quantile(samples_in_cluster, 0.5 - 0.5*c, dim=0)
                    up_curve = torch.quantile(samples_in_cluster, 0.5 + 0.5*c, dim=0)
                    c_interval = torch.stack([down_curve, up_curve], dim = 0) #[2, n_features]
                    confidence_intervals.append(c_interval)
                confidence_intervals = torch.stack(confidence_intervals, dim = 0) #[C, 2, n_features]

            else:
                median_curve = torch.zeros(n_features).to(samples.device)
                confidence_intervals = torch.zeros(len(confidence), 2, n_features).to(samples.device)
            num_in_mode.append(assignment.sum())
            mode_median.append(median_curve)
            mode_interval.append(confidence_intervals)
        num_in_mode = torch.tensor(num_in_mode).to(samples.device)
        mode_order = torch.argsort(num_in_mode, descending=True)
        num_in_mode = num_in_mode[mode_order]
        mode_median = torch.stack(mode_median, dim=0)[mode_order, :]  #[K, n_features]
        mode_interval = torch.stack(mode_interval, dim=0)[mode_order, :, :, :] #[K, C, 2, n_features]

        batch_num_in_mode.append(num_in_mode)
        batch_mode_median.append(mode_median)
        batch_confidence_interval.append(mode_interval)

    batch_num_in_mode = torch.stack(batch_num_in_mode, dim = 0)  #[batch_size, K]
    batch_mode_median = torch.stack(batch_mode_median, dim = 0)  #[batch_size, K, n_features]
    batch_confidence_interval = torch.stack(batch_confidence_interval, dim = 0) #[batch_size, K, C, 2, n_features]

    return batch_num_in_mode, batch_mode_median, batch_confidence_interval
