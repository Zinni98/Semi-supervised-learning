# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import torch 
import torch.nn as nn 
from torch.nn import functional as F


def smooth_targets(logits, targets, smoothing=0.1):
    """
    label smoothing
    """
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (logits.shape[-1] - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), (1 - smoothing))
    return true_dist


def ce_loss(logits, targets, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.

    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


def consistency_loss(logits, targets, name='ce', mask=None):
    """
    wrapper for consistency regularization loss in semi-supervised learning.

    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()


def contrastive_domain_discrepancy(lb_phis, ulb_phis, lb_labels, ulb_labels, classes, beta):
    def rbf_kernel(X, Y=None, gamma=None):
        if Y is None:
            Y = torch.clone(X)
        if gamma is None:
            gamma = 1.0 / X.size(1)
        K = torch.cdist(X, Y, compute_mode="use_mm_for_euclid_dist").square()
        K = torch.mul(K, -gamma)
        return torch.exp(K)  # exponentiate K in-place

    def D_cdd(phi_source, y_source, phi_target, y_target, classes):
        num_classes = len(classes)
        
        intra_sum = [domain_discrepancy(c, c, phi_source, y_source, phi_target, y_target)  
                     for c in classes]
 
        intra = torch.stack(intra_sum, dim=0).cuda().mean()

        inter_sum = [domain_discrepancy(c, c_prime, phi_source, y_source, phi_target, y_target) 
                     for c in classes 
                        for c_prime in classes 
                            if c != c_prime]
        inter = torch.stack(inter_sum, dim=0).cuda().mean()

        return torch.sub(intra, inter)

    def domain_discrepancy(c, c_prime, phi_source, y_source, phi_target, y_target):
        intra = torch.add(
            e(c, c, phi_source, y_source, phi_source, y_source),
            e(c_prime, c_prime, phi_target, y_target, phi_target, y_target)
        )
        inter = torch.mul(-2, e(c, c_prime, phi_source, y_source, phi_target, y_target))
        return torch.add(intra, inter)

    def e(c, c_prime, phi, y, phi_prime, y_prime):
        #kernel_covariance = rbf_kernel(phi, phi_prime)
        kernel_covariance = torch.cosine_similarity(phi, phi_prime)
        A = (y == c).unsqueeze(1).expand(-1, y_prime.size(0))
        B = (y_prime == c_prime).unsqueeze(0)
        mask = (A & B)
        masked = kernel_covariance*mask
        similarity = torch.sum(masked)
        count = torch.count_nonzero(masked)

        return torch.div(similarity, count) if count > 0 else torch.tensor(0.).cuda()

    return torch.mul(beta, D_cdd(lb_phis, lb_labels, ulb_phis, ulb_labels, classes))