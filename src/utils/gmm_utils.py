import torch
import torch.distributions as D


def gmm_params_to_dist(mean, std, prob):
    """
    Helper function that returns a torch distribution object for a
    Gaussian Mixture Model.
    :param mean: (batch_size, target_dim, n_components)
    :param std: (batch_size, target_dim, n_components)
    :param prob: (batch_size, target_dim, n_components)
    :return: torch distribution object
    """
    std = std.clip(min=0.01)
    comp = D.Normal(mean.squeeze(-1), std.squeeze(-1))

    assert len(prob.shape) == 3

    if prob.shape[-1] > 1:
        # Using a mixture of gaussians
        mix = D.Categorical(probs=prob)
        gmm = D.MixtureSameFamily(mix, comp)
        return gmm
    else:
        return comp


def gmm_cdf(target, mean, std, prob):
    """
    :param target: (batch_size, target_dim) or (label_queries, batch_size, target_dim)
    :param mean: (batch_size, target_dim, n_components)
    :param std: (batch_size, target_dim, n_components)
    :param prob: (batch_size, target_dim, n_components)
    :return: (batch_size) or (batch_size, label_queries) cdf
    """
    dist = gmm_params_to_dist(mean, std, prob)
    cdfs = dist.cdf(target)  # (batch_size, target_dim) or (label_queries, batch_size, target_dim)
    return torch.prod(cdfs, dim=-1)

def gmm_cdf_dist(target, dist):
    """
    :param target: (batch_size, target_dim) or (label_queries, batch_size, target_dim)
    :param dist: torch distribution object
    :return: (batch_size) or (batch_size, label_queries) cdf
    """
    cdfs = dist.cdf(target).view(-1,1)  # (batch_size, target_dim) or (label_queries, batch_size, target_dim)
    return torch.prod(cdfs, dim=-1)


def gmm_nll(target, mean, std, prob, reduce=True):
    """
    :param target: (batch_size, target_dim) or (label_queries, batch_size, target_dim)
    :param mean: (batch_size, target_dim, n_components)
    :param std: (batch_size, target_dim, n_components)
    :param prob: (batch_size, target_dim, n_components)
    :return: (batch_size) or (batch_size, label_queries) nll
    """
    dist = gmm_params_to_dist(mean, std, prob)
    log_prob = dist.log_prob(target)  # (batch_size, target_dim) or (label_queries, batch_size, target_dim)
    log_prob = log_prob.sum(dim=-1)  # (batch_size,) or (label_queries, batch_size)
    if reduce:
        log_prob = log_prob.sum(dim=-1)  # (1,) or (label_queries,)
    return - log_prob

def gmm_nll_dist(target, dist, reduce=True):
    """
    :param target: (batch_size, target_dim) or (label_queries, batch_size, target_dim)
    :param dist: torch distribution object
    :return: (batch_size) or (batch_size, label_queries) nll
    """
    log_prob = dist.log_prob(target).view(-1,1)  # (batch_size, target_dim) or (label_queries, batch_size, target_dim)
    log_prob = log_prob.sum(dim=-1)  # (batch_size,) or (label_queries, batch_size)
    if reduce:
        log_prob = log_prob.sum(dim=-1)  # (1,) or (label_queries,)
    return - log_prob


def gmm_nll_dummy_x(x, target, preds, reduce=True, use_dist=False):
    """
    Wrapper for gmm_nll function call. Used to compute the Negative Log-Likelihood
    for a Gaussian Mixture Model.
    """
    if use_dist:
        return gmm_nll_dist(preds)
    else:
        mean, std, prob = preds
        return gmm_nll(target, mean, std, prob, reduce=reduce)


def gmm_likelihood(target, mean, std, prob):
    """
    :param target: (batch_size, target_dim) or (label_queries, batch_size, target_dim)
    :param mean: (batch_size, target_dim, n_components)
    :param std: (batch_size, target_dim, n_components)
    :param prob: (batch_size, target_dim, n_components)
    :return: (batch_size) or (batch_size, label_queries)
    """
    dist = gmm_params_to_dist(mean, std, prob)
    log_prob = dist.log_prob(target)  # (batch_size, target_dim) or (label_queries, batch_size, target_dim)
    prob = log_prob.sum(dim=-1).exp()  # (batch_size,) or (label_queries, batch_size)
    return prob

def gmm_likelihood_dist(target, dist):
    """
    :param target: (batch_size, target_dim) or (label_queries, batch_size, target_dim)
    :param dist: torch distribution object
    :return: (batch_size) or (batch_size, label_queries)
    """
    log_prob = dist.log_prob(target).view(-1,1)  # (batch_size, target_dim) or (label_queries, batch_size, target_dim)
    prob = log_prob.sum(dim=-1).exp()  # (batch_size,) or (label_queries, batch_size)
    return prob