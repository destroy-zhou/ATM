import torch

def get_statistics(x_seq):
    '''
        x: time series with shape [batch_size, ts_length]
    '''
    mean = torch.mean(x_seq, dim=-1).detach()
    stdev = torch.sqrt(torch.var(x_seq, dim=-1, correction=0)).detach()
    #too small stdev will cause nan, set to 1
    stdev[stdev < 10*torch.finfo(stdev.dtype).eps] = 1.0

    return mean, stdev

def normalize(x_seq, mean, stdev):
    '''
    x_seq: [batch_size, ts_length]
    mean: [batch_size]
    stdev: [batch_size]
    '''

    x_seq = x_seq - mean[..., None]
    x_seq = x_seq / stdev[..., None]

    return x_seq

def denormalize(x_seq, mean, stdev):
    '''
    x_seq: [batch_size, (num_samples), ts_length]
    mean: [batch_size]
    stdev: [batch_size]
    '''
    while (len(mean.shape) < len(x_seq.shape)):
        mean = mean[..., None]
        stdev = stdev[..., None]

    x_seq = x_seq * stdev
    x_seq = x_seq + mean

    return x_seq
