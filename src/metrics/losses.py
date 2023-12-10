import torch
from torch import nn
import torch.nn.functional as F
from src.utils import gmm_nll_dummy_x, gmm_params_to_dist, get_bandwidth, check_input, get_ece_kde
from typing import Optional, Dict, List
from scipy.special import erf
import numpy as np
import time

def rbf_kernel(u: torch.Tensor, v: torch.Tensor, bandwidth=1):
    diff_norm_mat = torch.norm(u.unsqueeze(1) - v, dim=2).square()
    return torch.exp(- diff_norm_mat / bandwidth)


def quadrant_partition_kernel(u: torch.Tensor, v: torch.Tensor):
    raise NotImplementedError()


def norm_partition_kernel(u: torch.Tensor, v: torch.Tensor):
    raise NotImplementedError()

def tanh_kernel(u: torch.Tensor, v: torch.Tensor, bandwidth=1):
    out = torch.tanh(v) * torch.tanh(u).unsqueeze(1) # N x N x 1 x num_samples
    return out.squeeze(2)

kernel_funs = {"rbf": rbf_kernel,
               "partition_quadrant": quadrant_partition_kernel,
               "partition_norm": norm_partition_kernel,
               "tanh": tanh_kernel}

VALID_OPERANDS = ['x', 'y', 'p', 'coords']

class GMMKernelLoss:
    """
        MMD loss function for regression tasks using a Gaussian Mixture Model (GMM).
        Allows for distribution matching by specifying operands and kernel functions. 
        For example, by passing in operands = {'x':'rbf', 'y':'rbf'}, it will recover
        individual calibration by matching X and Y using universal rbf kernels.
        `num_samples` is used to marginalize over randomness of forecaster.
        `kwargs` specifies the parameters of the kernel functions, such as bandwidth
    """
    def __init__(self,
                 operands: Dict[str, str] = {'x': "rbf", 'p': "rbf"},
                 scalers: Optional[Dict] = None,
                 num_samples: Optional[int] = 10,
                 **kwargs):

        assert all([op in VALID_OPERANDS for op in operands.keys()])

        if scalers is None:
            scalers = {op: 1. for op in operands.keys()}
        else:
            assert all(op in scalers for op in operands.keys())

        self.kernel_fun = {op: kernel_funs[kernel] for op, kernel in operands.items()}
        self.operands = list(operands.keys())
        self.scalers = scalers
        self.num_samples = num_samples
        self.kwargs = kwargs

    def __call__(self, x, y, preds, num_true: Optional[int] = None, use_dist=False, verbose=False):
        if use_dist:
            dist = preds
        else:
            mean, std, prob = preds
            if 'y' in self.operands:
                # Check that number of components is 1 to ensure differentiable sampling.
                # Can use Gumbel-Softmax to allow for more than one component.
                assert prob.shape[-1] == 1
            dist = gmm_params_to_dist(mean, std, prob)

        # If number of samples from true distribution is not specified
        #  set equal to number of samples from predicted distribution
        num_pred = x.shape[0]
        if num_true is None:
            num_true = num_pred
            true_idxs = torch.arange(0, num_pred)
        else:
            print('WARNING: Currently, there are concerns that using mean_no_diag to cancel non-iid terms no longer works if num_true is passed in.')
            true_idxs = np.random.choice(num_pred, num_true, replace=False)

        kernel_out = None
        loss_mats = [None for i in range(3)]

        for op in self.operands:
            scaler = self.scalers[op]
            if op == 'coords':
                coords = x[:, :3]
                op_pred = coords
                op_true = coords[true_idxs]

                loss_mat = scaler * self.kernel_fun[op](op_pred, op_pred, **self.kwargs)
                loss_mat2 = scaler * self.kernel_fun[op](op_pred, op_true, **self.kwargs)
                loss_mat3 = scaler * self.kernel_fun[op](op_true, op_true, **self.kwargs)
            elif op == 'x':
                op_pred = x
                op_true = x[true_idxs]

                # This is only true for tabular data. For example, multi-channel images will have 4D batches for x.
                assert op_pred.dim() == 2
                assert op_true.dim() == 2

                loss_mat = scaler * self.kernel_fun[op](op_pred, op_pred, **self.kwargs)
                loss_mat2 = scaler * self.kernel_fun[op](op_pred, op_true, **self.kwargs)
                loss_mat3 = scaler * self.kernel_fun[op](op_true, op_true, **self.kwargs)

            elif op == 'y':
                y_pred = dist.rsample((self.num_samples,)).permute(1,2,0).to(x.device)
                op_pred = y_pred
                op_true = y[true_idxs].view(-1, 1)

                loss_mat = scaler * self.kernel_fun[op](op_pred, op_pred, **self.kwargs).mean(-1)
                loss_mat2 = scaler * self.kernel_fun[op](op_pred, op_true.unsqueeze(-1), **self.kwargs).mean(-1)
                loss_mat3 = scaler * self.kernel_fun[op](op_true, op_true, **self.kwargs)
            elif op == "p":
                p_pred = dist.cdf(y).view(-1, 1)
                p_true = torch.rand(num_true, device=x.device).view(-1, 1)
                op_pred = p_pred
                op_true = p_true

                loss_mat = scaler * self.kernel_fun[op](op_pred, op_pred, **self.kwargs)
                loss_mat2 = scaler * self.kernel_fun[op](op_pred, op_true, **self.kwargs)
                loss_mat3 = scaler * self.kernel_fun[op](op_true, op_true, **self.kwargs)

            if verbose:
                print(f'op = {op}, op_pred.device = {op_pred.device}, op_true.device = {op_true.device}')
            
            for i, value in enumerate([loss_mat, loss_mat2, loss_mat3]):
                if loss_mats[i] is None:
                    loss_mats[i] = value
                else:
                    loss_mats[i] =  loss_mats[i] * value

        kernel_out = mean_no_diag(loss_mats[0]) - 2 * loss_mats[1].mean() + mean_no_diag(loss_mats[2])

        return kernel_out


def mean_no_diag(A):
    assert A.dim() == 2 and A.shape[0] == A.shape[1]
    n = A.shape[0]
    A = A - torch.eye(n).to(A.device) * A.diag()
    return A.sum() / (n * (n - 1))


def GMMNLLLoss(**kwargs):
    return gmm_nll_dummy_x


class GMMMixedLoss:
    """
        Mixed loss function (MMD + NLL) for regression tasks using a Gaussian Mixture Model (GMM).
        `loss_scalers` determines the mixture weight between MMD and NLL.
    """
    def __init__(self, loss_scalers: Optional[Dict] = None, **kwargs):
        if loss_scalers is None:
            loss_scalers = {"nll": .01, "mmd": 1}
        else:
            assert set(loss_scalers.keys()) == {"nll", "mmd"}
        self.loss_scalers = loss_scalers
        self.nll = GMMNLLLoss()
        self.mmd = GMMKernelLoss(**kwargs)

    def __call__(self, x, y, preds, use_dist=False):
        return self.loss_scalers["nll"] * self.nll(x, y, preds, use_dist=use_dist) + self.loss_scalers["mmd"] * self.mmd(x, y, preds, use_dist=use_dist)


#### Classification loss functions #####
class ClassificationKernelLoss:
    """
        MMD loss function for classification tasks.
        Allows for distribution matching by specifying operands and kernel functions. 
        `scalers` and `bandwidths` are the parameters of the kernel functions.
    """
    def __init__(self,
                 operands: Dict[str, str] = {'x': "rbf", 'y': "rbf"},
                 scalers: Optional[Dict] = None,
                 bandwidths: Optional[Dict] = {'x': 10.0, 'y': 0.1}):

        assert all([op in VALID_OPERANDS for op in operands.keys()])

        if scalers is None:
            scalers = {op: 1. for op in operands.keys()}
        else:
            assert all(op in scalers for op in operands.keys())

        self.kernel_fun = {op: kernel_funs[kernel] for op, kernel in operands.items()}
        self.operands = list(operands.keys())
        self.scalers = scalers
        self.bandwidths = bandwidths

    def __call__(self, x, y, logits, verbose=False):
        kernel_out = None
        loss_mats = [None for i in range(3)]

        for op in self.operands:
            scaler = self.scalers[op]
            bandwidth = self.bandwidths[op]
            if op == 'x':
                # This is only true for tabular data. For example, multi-channel images will have 4D batches for x.
                assert x.dim() == 2
                loss_mat = loss_mat2 = loss_mat3 = scaler * self.kernel_fun[op](x, x, bandwidth)
            elif op == 'y':
                # Computes MMD loss for classification (See Section 4.1 of paper)
                num_classes = logits.shape[-1]
                y_all = torch.eye(num_classes).to(logits.device)
                k_yy = self.kernel_fun[op](y_all, y_all, bandwidth)
                q_y = F.softmax(logits, dim=-1)
                q_yy = torch.einsum('ic,jd->ijcd', q_y, q_y)
                total_yy = q_yy * k_yy.unsqueeze(0)

                k_yj = k_yy[:,y].T
                total_yj = torch.einsum('ic,jc->ijc', q_y, k_yj)
                y_one_hot = F.one_hot(y, num_classes=num_classes).float()

                loss_mat = scaler * total_yy.sum(dim=(2,3))
                loss_mat2 = scaler * total_yj.sum(-1)
                loss_mat3 = scaler * self.kernel_fun[op](y_one_hot, y_one_hot, bandwidth)
            else:
                assert False, f"When running classification, operands must be x and y. Got operand {op} instead."
            
            for i, value in enumerate([loss_mat, loss_mat2, loss_mat3]):
                if loss_mats[i] is None:
                    loss_mats[i] = value
                else:
                    loss_mats[i] =  loss_mats[i] * value

        kernel_out = mean_no_diag(loss_mats[0]) - 2 * mean_no_diag(loss_mats[1]) + mean_no_diag(loss_mats[2])

        return kernel_out

class ClassificationCELoss:
    """
        Cross-entropy loss for classification.
    """
    def __init__(self, **kwargs):
        self.loss = torch.nn.CrossEntropyLoss(**kwargs)

    def __call__(self, x, y, logits):
        return self.loss(logits, y)

class ClassificationMixedLoss:
    """
        Mixed loss function (MMD + NLL) for classification.
        `loss_scalers` determines the mixture weight between MMD and NLL.
    """
    def __init__(self, loss_scalers: Optional[Dict] = None, **kwargs):
        if loss_scalers is None:
            loss_scalers = {"nll": .01, "mmd": 1}
        else:
            assert set(loss_scalers.keys()) == {"nll", "mmd"}
        self.loss_scalers = loss_scalers
        self.nll = torch.nn.CrossEntropyLoss()
        self.mmd = ClassificationKernelLoss(**kwargs)

    def __call__(self, x, y, logits):
        return self.loss_scalers["nll"] * self.nll(logits, y) + self.loss_scalers["mmd"] * self.mmd(x, y, logits)

class MMCEMixedLoss:
    """
        MMCE loss from paper "Trainable Calibration Measures For Neural Networks From Kernel Mean Embeddings"
        Code based on official author implementation: https://github.com/aviralkumar2907/MMCE
    """
    def __init__(self, loss_scalers: Optional[Dict] = None, **kwargs):
        if loss_scalers is None:
            loss_scalers = {"nll": 1.0, "mmce": 4.0}
        else:
            assert set(loss_scalers.keys()) == {"nll", "mmce"}
        self.loss_scalers = loss_scalers
        self.nll = torch.nn.CrossEntropyLoss()
        self.mmce = self.torch_mmce_w_loss

    def torch_mmce_w_loss(self, logits, correct_labels):
        """Function to compute the MMCE_w loss."""

        predicted_probs = F.softmax(logits, dim=-1)
        range_index = torch.arange(0, predicted_probs.shape[0]).to(dtype=torch.long).unsqueeze(1)
        predicted_labels = torch.argmax(predicted_probs, dim=1)

        gather_index = torch.concat([range_index, predicted_labels.unsqueeze(1)], dim=1)
        predicted_probs, _ = predicted_probs.max(1)
        correct_mask = torch.where(correct_labels == predicted_labels, 1, 0)
        sigma = 0.2

        def torch_kernel(matrix):
            """Kernel was taken to be a laplacian kernel with sigma = 0.4."""
            return torch.exp(-1.0*torch.abs(matrix[:, :, 0] - matrix[:, :, 1])/(2*0.2))

        k = torch.sum(correct_mask)
        k_p = torch.sum(1 - correct_mask)

        cond_k = torch.where(k == 0, 0, 1)
        cond_k_p = torch.where(k_p == 0, 0, 1)
        k = torch.clamp(k, min=1)*cond_k*cond_k_p + (1 - cond_k*cond_k_p)*2 
        k_p = torch.clamp(k_p, min=1)*cond_k_p*cond_k + ((1 - cond_k_p*cond_k)*
                                                    (correct_mask.shape[0] - 2))

        correct_prob, _ = torch.topk(predicted_probs*correct_mask, k)
        incorrect_prob, _ = torch.topk(predicted_probs*(1 - correct_mask), k_p)
        
        def get_pairs(tensor1, tensor2):
            correct_prob_tiled = torch.unsqueeze(torch.tile(torch.unsqueeze(tensor1, 1),
                            [1, tensor1.shape[0]]), 2)
            incorrect_prob_tiled = torch.unsqueeze(torch.tile(torch.unsqueeze(tensor2, 1),
                            [1, tensor2.shape[0]]), 2)
            correct_prob_pairs = torch.concat([correct_prob_tiled,
                            torch.permute(correct_prob_tiled, [1, 0, 2])],
                            axis=2)
            incorrect_prob_pairs = torch.concat([incorrect_prob_tiled,
                        torch.permute(incorrect_prob_tiled, [1, 0, 2])],
                        axis=2)
            correct_prob_tiled_1 = torch.unsqueeze(torch.tile(torch.unsqueeze(tensor1, 1),
                                [1, tensor2.shape[0]]), 2)
            incorrect_prob_tiled_1 = torch.unsqueeze(torch.tile(torch.unsqueeze(tensor2, 1),
                                [1, tensor1.shape[0]]), 2)
            correct_incorrect_pairs = torch.concat([correct_prob_tiled_1,
                        torch.permute(incorrect_prob_tiled_1, [1, 0, 2])],
                        axis=2)
            return correct_prob_pairs, incorrect_prob_pairs, correct_incorrect_pairs
        
        correct_prob_pairs, incorrect_prob_pairs,\
                    correct_incorrect_pairs = get_pairs(correct_prob, incorrect_prob)
        correct_kernel = torch_kernel(correct_prob_pairs)
        incorrect_kernel = torch_kernel(incorrect_prob_pairs)
        correct_incorrect_kernel = torch_kernel(correct_incorrect_pairs)  
        sampling_weights_correct = torch.matmul(torch.unsqueeze(1.0 - correct_prob, 1),
                                torch.unsqueeze(1.0 - correct_prob, 1).T)
        correct_correct_vals = torch.mean(correct_kernel * sampling_weights_correct)
        sampling_weights_incorrect = torch.matmul(torch.unsqueeze(incorrect_prob, 1),
                                torch.unsqueeze(incorrect_prob, 1).T)
        incorrect_incorrect_vals = torch.mean(incorrect_kernel * sampling_weights_incorrect)
        sampling_correct_incorrect = torch.matmul(torch.unsqueeze(1.0 - correct_prob, 1),
                                torch.unsqueeze(incorrect_prob, 1).T)
        correct_incorrect_vals = torch.mean(correct_incorrect_kernel * sampling_correct_incorrect)
        correct_denom = torch.sum(1.0 - correct_prob)
        incorrect_denom = torch.sum(incorrect_prob)
        m = torch.sum(correct_mask)
        n = torch.sum(1.0 - correct_mask)
        mmd_error = 1.0/(m*m + 1e-5) * torch.sum(correct_correct_vals) 
        mmd_error += 1.0/(n*n + 1e-5) * torch.sum(incorrect_incorrect_vals)
        mmd_error -= 2.0/(m*n + 1e-5) * torch.sum(correct_incorrect_vals)

        return torch.clamp((cond_k*cond_k_p).detach()*torch.sqrt(mmd_error + 1e-10), min=0.0)

    def __call__(self, x, y, logits):
        return self.loss_scalers["nll"] * self.nll(logits, y) + self.loss_scalers["mmce"] * self.mmce(logits, y)

class ECEKDEMixedLoss:
    """ ECE KDE loss from paper "A Consistent and Differentiable Lp Canonical Calibration Error Estimator", published in NeurIPS 2022.
        Code based on official author implementation: https://github.com/tpopordanoska/ece-kde
    """
    def __init__(self, loss_scalers: Optional[Dict] = None, bandwidth: Optional[float] = None, p: Optional[int] = 1, mc_type: Optional[str] = 'canonical', **kwargs):
        """
        :param bandwidth: The bandwidth of the kernel
        :param p: The p-norm. Typically, p=1 or p=2
        :param mc_type: The type of multiclass calibration: canonical, marginal or top_label
        """
        if loss_scalers is None:
            loss_scalers = {"nll": 1.0, "ece_kde": 0.001}
        else:
            assert set(loss_scalers.keys()) == {"nll", "ece_kde"}
        self.loss_scalers = loss_scalers
        self.nll = torch.nn.CrossEntropyLoss()
        self.ece_kde = self.ece_kde_loss

        self.bandwidth = bandwidth
        self.p = p
        self.mc_type = mc_type

    def ece_kde_loss(self, logits, y):
        """Function to compute the ECE KDE loss from "A Consistent and Differentiable Lp Canonical Calibration Error Estimator" paper, published in NeurIPS 2022.

        :param logits: The vector containing the logits shape [num_samples, num_classes]
        :param y: The vector containing the labels, shape [num_samples]

        :return: An estimate of Lp calibration error
        """
        probs = F.softmax(logits, dim=-1)

        if self.bandwidth is None:
            bandwidth = max(get_bandwidth(probs), 1e-5)
        else:
            bandwidth = self.bandwidth

        return get_ece_kde(probs, y, bandwidth, self.p, self.mc_type)

    def __call__(self, x, y, logits):
        return self.loss_scalers["nll"] * self.nll(logits, y) + self.loss_scalers["ece_kde"] * self.ece_kde_loss(logits, y)