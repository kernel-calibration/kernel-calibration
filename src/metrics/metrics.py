from torchmetrics import Metric
import torch
from src.utils import gmm_nll, gmm_cdf, gmm_nll_dist, gmm_cdf_dist, gmm_params_to_dist
import numpy as np
from src.metrics.losses import GMMKernelLoss, ClassificationKernelLoss
import src.metrics.decision_losses as decision_losses
from typing import List
from torchuq.transform.conformal import DistributionConformal
import torch.nn.functional as F


class GMMNegativeLogLikelihood(Metric):
    def __init__(self, dist_sync_on_step=False, use_dist=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.use_dist = use_dist
        self.add_state("nll_total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.use_dist:
            nll = gmm_nll_dist(target, preds, reduce=False)
        else:
            mean, std, prob = preds
            nll = gmm_nll(target, mean, std, prob, reduce=False)

        self.nll_total += nll.sum()
        self.count += target.shape[0]

    def compute(self):
        return self.nll_total.float() / self.count


# 0 is perfect calibration, 1 is worst possible
class GMMCalibrationError(Metric):
    def __init__(self, dist_sync_on_step=False, bins: int = 20, use_dist=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("bin_counts", default=torch.zeros(bins), dist_reduce_fx="sum")
        self.bins = bins
        self.use_dist = use_dist

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.use_dist:
            cdf = gmm_cdf_dist(target, preds)
        else:
            mean, std, prob = preds
            cdf = gmm_cdf(target, mean, std, prob)

        bin_ids = (cdf * self.bins).round().clamp(max=self.bins - 1)
        bin_ids, counts = torch.unique(bin_ids, return_counts=True)
        bin_counts = torch.zeros(self.bins, device=self.bin_counts.device)
        bin_counts[bin_ids.type(torch.long)] = counts.type(torch.float)

        self.bin_counts += bin_counts

    def compute(self):
        target = torch.full_like(self.bin_counts, 1 / self.bins)
        actual = self.bin_counts / self.bin_counts.sum()
        return torch.abs(actual - target).sum() / 2


class GMMKernelCalibrationError(Metric):
    def __init__(self, dist_sync_on_step=False, use_dist=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("kcal_total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.), dist_reduce_fx="sum")

        self.kwargs = kwargs
        self.use_dist = use_dist
        self.kcal_func = GMMKernelLoss(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor, input: torch.Tensor, verbose=False):
        kcal = self.kcal_func(input, target, preds, use_dist=self.use_dist, verbose=verbose)

        self.kcal_total += kcal
        self.count += 1

    def compute(self):
        return self.kcal_total.float() / self.count

class ClassificationKernelCalibrationError(Metric):
    def __init__(self, dist_sync_on_step=False, **kwargs):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("kcal_total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.kcal_func = ClassificationKernelLoss(**kwargs)

    def update(self, preds: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor, verbose=False):
        kcal = self.kcal_func(inputs, target, preds, verbose=verbose)

        self.kcal_total += kcal
        self.count += 1

    def compute(self):
        return self.kcal_total.float() / self.count

class DecisionCalibrationError(Metric):
    def __init__(self, dist_sync_on_step=False, loss_fn_cls: str = "", actions: List = [], metric='L2', num_samples: int = 10, use_dist=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.loss_fn = getattr(decision_losses, loss_fn_cls)
        self.actions = actions
        self.metric = metric
        self.num_samples = num_samples
        self.use_dist = use_dist

        self.add_state("dcal_total", default=torch.tensor(0.), dist_reduce_fx="sum")
        if metric == 'mean':
            self.add_state("dcal_count", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        E_p = {a: None for a in self.actions}
        E_q = {a: None for a in self.actions}

        if self.use_dist:
            dists = preds
        else:
            mean, std, prob = preds
            dists = gmm_params_to_dist(mean, std, prob)

        if isinstance(dists, DistributionConformal):
            target_samples = dists.sample_n(self.num_samples).view(-1 ,1).to(target.device)
        else:
            target_samples = dists.sample((self.num_samples,)).flatten(0,1).to(target.device)

        for a in self.actions:
            # Estimate E_p[loss_fn(a,y)]
            E_p[a] = self.loss_fn(a, target).mean()
            
            # Estimate E_q[loss_fn(a,y)]
            E_q[a] = self.loss_fn(a, target_samples).mean()
        
        E_p_vals = torch.stack(list(E_p.values()))
        E_q_vals = torch.stack(list(E_q.values()))

        if self.metric == 'L2':
            self.dcal_total += ((E_p_vals - E_q_vals) ** 2).sum()
            dcal = self.dcal_total**0.5
        elif self.metric == 'mean':
            self.dcal_total += (E_p_vals - E_q_vals).sum()
            self.dcal_count += target.shape[0]
        else:
            raise NotImplementedError()
    
    def compute(self):
        if self.metric == 'L2':
            dcal = self.dcal_total**0.5
        elif self.metric == 'mean':
            dcal = self.dcal_total / self.dcal_count
        else:
            raise NotImplementedError()
        
        return dcal

#### Classification metrics

class ShannonEntropyError(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("entropy_total", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, is_dist=False):
        p = logits if is_dist else F.softmax(logits, dim=-1)

        self.entropy_total += torch.sum(- p * torch.log(p))
        self.count += logits.shape[0]

    def compute(self):
        return self.entropy_total.float() / self.count.float()