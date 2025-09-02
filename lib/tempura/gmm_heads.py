import torch
import torch.nn as nn


class GMM_head(nn.Module):
    """Gaussian Mixture Model head for uncertainty estimation in Tempura.

    Implements a GMM-based classification head that models uncertainty
    through multiple Gaussian components with learnable means, variances,
    and mixture weights.

    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """

    def __init__(self, hid_dim, num_classes, rel_type=None, k=4):
        """Initialize the GMM head.

        :param hid_dim: Hidden dimension size
        :type hid_dim: int
        :param num_classes: Number of output classes
        :type num_classes: int
        :param rel_type: Type of relation (affects activation function), defaults to None
        :type rel_type: str, optional
        :param k: Number of Gaussian mixture components, defaults to 4
        :type k: int, optional
        :return: None
        :rtype: None
        """
        super(GMM_head, self).__init__()
        self.k = k
        self.num_classes = num_classes
        self.heads = nn.ModuleDict()
        self.rel_type = rel_type
        for i in range(k):
            self.heads.update(
                {
                    "mu_" + str(i + 1): nn.Linear(hid_dim, num_classes),
                    "pi_" + str(i + 1): nn.Linear(hid_dim, 1),
                    "var_" + str(i + 1): nn.Linear(hid_dim, num_classes),
                }
            )
        if rel_type == "attention" or rel_type is None:
            self.activation = nn.Softmax(-1)
        else:
            self.activation = nn.Sigmoid()

    def _reset_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def uncertainty(self, conf_mu_k, conf_var_k, conf_pi_k_):
        """Compute epistemic and aleatoric uncertainty.

        :param conf_mu_k: Mean predictions for each mixture component
        :type conf_mu_k: dict
        :param conf_var_k: Variance predictions for each mixture component
        :type conf_var_k: dict
        :param conf_pi_k_: Mixture weights for each component
        :type conf_pi_k_: list
        :return: Tuple containing prediction, aleatoric uncertainty, and epistemic uncertainty
        :rtype: tuple
        """
        batch_conf_k = [conf_mu_k[str(i + 1)] for i in range(self.k)]
        new_conf = torch.zeros(batch_conf_k[0].shape).to(batch_conf_k[0].device)
        for i in range(self.k):
            new_conf = new_conf + self.activation(batch_conf_k[i]) * conf_pi_k_[i].view(
                -1, 1
            )

        al_uc = sum(
            [conf_var_k[str(i + 1)] * conf_pi_k_[i].view(-1, 1) for i in range(self.k)]
        )
        ep_uc = sum(
            [
                ((self.activation(batch_conf_k[i]) - new_conf) ** 2)
                * conf_pi_k_[i].view(-1, 1)
                for i in range(self.k)
            ]
        )

        return al_uc, ep_uc

    def forward(self, x, phase="train", unc=False):
        conf_var_k = {}
        conf_mu_k = {}
        conf_pi_k = {}

        for i in range(self.k):
            conf_mu_k[str(i + 1)] = self.heads["mu_" + str(i + 1)](x)
            conf_pi_k[str(i + 1)] = self.heads["pi_" + str(i + 1)](x)
            conf_var_k[str(i + 1)] = self.heads["var_" + str(i + 1)](x).sigmoid()

        conf_pi_all = torch.stack(
            [conf_pi_k[str(i + 1)].reshape(-1) for i in range(self.k)]
        ).transpose(0, 1)
        # (batch_size , K)
        conf_pi_all = (torch.softmax(conf_pi_all, dim=1)).transpose(0, 1).reshape(-1)
        # (batch_size*K, )
        conf_pi_k_ = torch.split(conf_pi_all, conf_pi_k["1"].reshape(-1).size(0), dim=0)
        # tuple of len K

        if unc:
            return self.uncertainty(conf_mu_k, conf_var_k, conf_pi_k_)

        rand_val_k = [
            torch.randn(conf_var_k[str(i + 1)].shape).to(conf_var_k[str(i + 1)].device)
            for i in range(self.k)
        ]
        if phase == "train":
            batch_conf_k = [
                (
                    conf_mu_k[str(i + 1)]
                    + torch.sqrt(conf_var_k[str(i + 1)]) * rand_val_k[i]
                )
                for i in range(self.k)
            ]
        else:
            if self.rel_type is not None:
                batch_conf_k = [conf_mu_k[str(i + 1)] for i in range(self.k)]
            else:
                batch_conf_k = [conf_mu_k[str(i + 1)][:, 1:] for i in range(self.k)]

        weighted_logits = torch.zeros(batch_conf_k[0].shape).to(batch_conf_k[0].device)
        for i in range(self.k):
            weighted_logits = weighted_logits + self.activation(
                batch_conf_k[i]
            ) * conf_pi_k_[i].view(-1, 1)

        return weighted_logits
