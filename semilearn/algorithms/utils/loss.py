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
        inter = torch.mul(-2, e(c, c_prime, phi_source,
                          y_source, phi_target, y_target))
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


class CDD(object):
    def __init__(self, num_layers, num_classes, kernel_num=(5, 5), kernel_mul=(2, 2),
                 intra_only=False, **kwargs):

        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.num_classes = num_classes
        self.intra_only = intra_only or (self.num_classes == 1)
        self.num_layers = num_layers

    def split_classwise(self, dist, nums):
        num_classes = len(nums)
        start = end = 0
        dist_list = []
        for c in range(num_classes):
            start = end
            end = start + nums[c]
            dist_c = dist[start:end, start:end]
            dist_list += [dist_c]
        return dist_list

    def gamma_estimation(self, dist):
        dist_sum = torch.sum(dist['ss']) + torch.sum(dist['tt']) + \
            2 * torch.sum(dist['st'])

        bs_S = dist['ss'].size(0)
        bs_T = dist['tt'].size(0)
        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T
        gamma = dist_sum.item() / N
        return gamma

    def patch_gamma_estimation(self, nums_S, nums_T, dist):
        assert(len(nums_S) == len(nums_T))
        num_classes = len(nums_S)

        patch = {}
        gammas = {}
        gammas['st'] = torch.zeros_like(dist['st'], requires_grad=False).cuda()
        gammas['ss'] = []
        gammas['tt'] = []
        for c in range(num_classes):
            gammas['ss'] += [torch.zeros([num_classes],
                                         requires_grad=False).cuda()]
            gammas['tt'] += [torch.zeros([num_classes],
                                         requires_grad=False).cuda()]

        source_start = source_end = 0
        for ns in range(num_classes):
            source_start = source_end
            source_end = source_start + nums_S[ns]
            patch['ss'] = dist['ss'][ns]

            target_start = target_end = 0
            for nt in range(num_classes):
                target_start = target_end
                target_end = target_start + nums_T[nt]
                patch['tt'] = dist['tt'][nt]

                patch['st'] = dist['st'].narrow(0, source_start,
                                                nums_S[ns]).narrow(1, target_start, nums_T[nt])

                gamma = self.gamma_estimation(patch)

                gammas['ss'][ns][nt] = gamma
                gammas['tt'][nt][ns] = gamma
                gammas['st'][source_start:source_end,
                             target_start:target_end] = gamma

        return gammas

    def compute_kernel_dist(self, dist, gamma, kernel_num, kernel_mul):
        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        gamma_list = [base_gamma * (kernel_mul**i) for i in range(kernel_num)]
        gamma_tensor = torch.stack(gamma_list, dim=0).cuda()

        eps = 1e-5
        gamma_mask = (gamma_tensor < eps).type(torch.cuda.FloatTensor)
        gamma_tensor = (1.0 - gamma_mask) * gamma_tensor + gamma_mask * eps
        gamma_tensor = gamma_tensor.detach()

        for i in range(len(gamma_tensor.size()) - len(dist.size())):
            dist = dist.unsqueeze(0)

        dist = dist / gamma_tensor
        upper_mask = (dist > 1e5).type(torch.cuda.FloatTensor).detach()
        lower_mask = (dist < 1e-5).type(torch.cuda.FloatTensor).detach()
        normal_mask = 1.0 - upper_mask - lower_mask
        dist = normal_mask * dist + upper_mask * 1e5 + lower_mask * 1e-5
        kernel_val = torch.sum(torch.exp(-1.0 * dist), dim=0)
        return kernel_val

    def kernel_layer_aggregation(self, dist_layers, gamma_layers, key, category=None):
        num_layers = self.num_layers
        kernel_dist = None
        for i in range(num_layers):

            dist = dist_layers[i][key] if category is None else \
                dist_layers[i][key][category]

            gamma = gamma_layers[i][key] if category is None else \
                gamma_layers[i][key][category]

            cur_kernel_num = self.kernel_num[i]
            cur_kernel_mul = self.kernel_mul[i]

            if kernel_dist is None:
                kernel_dist = self.compute_kernel_dist(dist,
                                                       gamma, cur_kernel_num, cur_kernel_mul)

                continue

            kernel_dist += self.compute_kernel_dist(dist, gamma,
                                                    cur_kernel_num, cur_kernel_mul)

        return kernel_dist

    def patch_mean(self, nums_row, nums_col, dist):
        assert(len(nums_row) == len(nums_col))
        num_classes = len(nums_row)

        mean_tensor = torch.zeros([num_classes, num_classes]).cuda()
        row_start = row_end = 0
        for row in range(num_classes):
            row_start = row_end
            row_end = row_start + nums_row[row]

            col_start = col_end = 0
            for col in range(num_classes):
                col_start = col_end
                col_end = col_start + nums_col[col]
                val = torch.mean(dist.narrow(0, row_start,
                                             nums_row[row]).narrow(1, col_start, nums_col[col]))
                mean_tensor[row, col] = val
        return mean_tensor

    def compute_paired_dist(self, A, B):
        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
        dist = (((A_expand - B_expand))**2).sum(2)
        return dist

    def forward(self, source, target, nums_S, nums_T):
        assert(len(nums_S) == len(nums_T)), \
            "The number of classes for source (%d) and target (%d) should be the same." \
            % (len(nums_S), len(nums_T))

        num_classes = len(nums_S)

        # compute the dist
        dist_layers = []
        gamma_layers = []

        for i in range(self.num_layers):

            cur_source = source[i]
            cur_target = target[i]

            dist = {}
            dist['ss'] = self.compute_paired_dist(cur_source, cur_source)
            dist['tt'] = self.compute_paired_dist(cur_target, cur_target)
            dist['st'] = self.compute_paired_dist(cur_source, cur_target)

            dist['ss'] = self.split_classwise(dist['ss'], nums_S)
            dist['tt'] = self.split_classwise(dist['tt'], nums_T)
            dist_layers += [dist]

            gamma_layers += [self.patch_gamma_estimation(nums_S, nums_T, dist)]

        # compute the kernel dist
        for i in range(self.num_layers):
            for c in range(num_classes):
                gamma_layers[i]['ss'][c] = gamma_layers[i]['ss'][c].view(
                    num_classes, 1, 1)
                gamma_layers[i]['tt'][c] = gamma_layers[i]['tt'][c].view(
                    num_classes, 1, 1)

        kernel_dist_st = self.kernel_layer_aggregation(
            dist_layers, gamma_layers, 'st')
        kernel_dist_st = self.patch_mean(nums_S, nums_T, kernel_dist_st)

        kernel_dist_ss = []
        kernel_dist_tt = []
        for c in range(num_classes):
            kernel_dist_ss += [torch.mean(self.kernel_layer_aggregation(dist_layers,
                                                                        gamma_layers, 'ss', c).view(num_classes, -1), dim=1)]
            kernel_dist_tt += [torch.mean(self.kernel_layer_aggregation(dist_layers,
                                                                        gamma_layers, 'tt', c).view(num_classes, -1), dim=1)]

        kernel_dist_ss = torch.stack(kernel_dist_ss, dim=0)
        kernel_dist_tt = torch.stack(kernel_dist_tt, dim=0).transpose(1, 0)

        mmds = kernel_dist_ss + kernel_dist_tt - 2 * kernel_dist_st
        intra_mmds = torch.diag(mmds, 0)
        intra = torch.sum(intra_mmds) / self.num_classes

        inter = None
        if not self.intra_only:
            inter_mask = ((torch.ones([num_classes, num_classes])
                           - torch.eye(num_classes)).type(torch.ByteTensor)).cuda()
            inter_mmds = torch.masked_select(mmds, inter_mask)
            inter = torch.sum(inter_mmds) / \
                (self.num_classes * (self.num_classes - 1))

        cdd = intra if inter is None else intra - inter
        return {'cdd': cdd, 'intra': intra, 'inter': inter}
