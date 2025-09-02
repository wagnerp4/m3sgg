from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning loss implementation.

    Based on the paper: https://arxiv.org/pdf/2004.11362.pdf
    Also supports unsupervised contrastive loss as used in SimCLR.

    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """

    def __init__(self, temperature=0.1, contrast_mode="all", base_temperature=0.07):
        """Initialize the supervised contrastive loss.

        :param temperature: Temperature parameter for scaling, defaults to 0.1
        :type temperature: float, optional
        :param contrast_mode: Contrast mode ('all' or 'one'), defaults to "all"
        :type contrast_mode: str, optional
        :param base_temperature: Base temperature for normalization, defaults to 0.07
        :type base_temperature: float, optional
        :return: None
        :rtype: None
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute contrastive loss for the model.

        If both labels and mask are None, it degenerates to SimCLR unsupervised loss.
        Reference: https://arxiv.org/pdf/2002.05709.pdf

        :param features: Hidden vector of shape [bsz, n_views, ...]
        :type features: torch.Tensor
        :param labels: Ground truth labels of shape [bsz], defaults to None
        :type labels: torch.Tensor, optional
        :param mask: Contrastive mask of shape [bsz, bsz], defaults to None
        :type mask: torch.Tensor, optional
        :return: Scalar loss value
        :rtype: torch.Tensor
        """
        device = features.device if features.is_cuda else torch.device("cpu")

        # if len(features.shape) < 3:
        #     raise ValueError('`features` needs to be [bsz, n_views, ...],'
        #                      'at least 3 dimensions are required')
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            lbl_mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # contrast_count = features.shape[1]
        contrast_feature = torch.nn.functional.normalize(features, dim=-1)
        # contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            # anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.ones_like(logits, device=device)
        logits_mask.fill_diagonal_(0)
        mask = lbl_mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / lbl_mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss


class EucNormLoss(nn.Module):
    def __init__(self):
        super(EucNormLoss, self).__init__()

    def forward(self, features, labels):
        labels = labels.contiguous().view(-1, 1)
        lbl_mask = torch.eq(labels, labels.T).float()
        feat = torch.nn.functional.normalize(features, p=2, dim=-1)
        anchor_dot_contrast = torch.cdist(feat, feat, p=2)

        loss = (anchor_dot_contrast * lbl_mask).relu()
        # print(loss.shape,anchor_dot_contrast.shape,lbl_mask.shape)
        loss = loss.sum(1) / lbl_mask.sum(1)
        loss = loss.mean()

        return loss
