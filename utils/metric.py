from torch import Tensor


def cross_entropy(p: Tensor, q: Tensor, eps: float=1e-16) -> Tensor:
    """Compute the negative cross-entropy between p and q, i.e., CE[p||q]"""
    return (p * (q.relu() + eps).log()).sum()


def kl_divergence(p: Tensor, q: Tensor, eps: float=1e-16) -> Tensor:
    """
    Compute the KL-divergence between p and q, i.e., KL[p||q].
    """
    return cross_entropy(p, p, eps) - cross_entropy(p, q, eps)


def nll_gaussian(preds: Tensor, target: Tensor, variance: float) -> Tensor:
    """
    Compute the negative log-likelihood of a Gaussian distribution with a fixed variance.

    Args:
        preds: [batch, steps, node, dim]
        target: [batch, steps, node, dim]
        variance: a fixed variance

    Return:
        negative log-likelihood: [steps, dim]
    """
    neg_log_p = ((preds - target) ** 2 / (2 * variance))
    return neg_log_p.sum() / (target.size(0) * target.size(2))
