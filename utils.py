import torch
import numpy as np
import scipy


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def hungarian_loss(predictions, targets, mask, pool):
    # predictions and targets shape :: (n, c, s)
    predictions = predictions[:,:mask,:]
    targets = targets[:,:mask,:]
    predictions = predictions.permute(0, 2, 1)
    targets = targets.permute(0, 2, 1)
    predictions, targets = outer(predictions, targets)
    # squared_error shape :: (n, s, s)
    squared_error = torch.sqrt((predictions - targets).pow(2).mean(1))
    squared_error_np = squared_error.detach().cpu().numpy()
    indices = pool.map(per_sample_hungarian_loss, squared_error_np)
    # print(indices)
    losses = [sample[row_idx, col_idx].mean() for sample, (row_idx, col_idx) in zip(squared_error, indices)]
    total_loss = torch.mean(torch.stack(list(losses)))
    return total_loss, indices[0][1]


def outer(a, b=None):
    if b is None:
        b = a
    size_a = tuple(a.size()) + (b.size()[-1],)
    size_b = tuple(b.size()) + (a.size()[-1],)
    a = a.unsqueeze(dim=-1).expand(*size_a)
    b = b.unsqueeze(dim=-2).expand(*size_b)
    return a, b


def per_sample_hungarian_loss(sample_np):
    sample_np[sample_np == np.inf] = 0
    row_idx, col_idx = scipy.optimize.linear_sum_assignment(sample_np)
    return row_idx, col_idx