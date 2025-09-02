import math

import torch
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """Implements AdamW algorithm with decoupled weight decay.

    The original Adam algorithm was proposed in `Adam: A Method for Stochastic Optimization`_.
    The AdamW variant was proposed in `Decoupled Weight Decay Regularization`_.

    :param params: Iterable of parameters to optimize or dicts defining parameter groups
    :type params: iterable
    :param lr: Learning rate, defaults to 1e-3
    :type lr: float, optional
    :param betas: Coefficients used for computing running averages of gradient and its square, defaults to (0.9, 0.999)
    :type betas: tuple, optional
    :param eps: Term added to the denominator to improve numerical stability, defaults to 1e-8
    :type eps: float, optional
    :param weight_decay: Weight decay coefficient, defaults to 1e-2
    :type weight_decay: float, optional
    :param amsgrad: Whether to use the AMSGrad variant, defaults to False
    :type amsgrad: bool, optional

    .. _Adam: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
    ):
        """Initialize the AdamW optimizer.

        :param params: Iterable of parameters to optimize
        :type params: iterable
        :param lr: Learning rate, defaults to 1e-3
        :type lr: float, optional
        :param betas: Coefficients for computing running averages, defaults to (0.9, 0.999)
        :type betas: tuple, optional
        :param eps: Term added to denominator for numerical stability, defaults to 1e-8
        :type eps: float, optional
        :param weight_decay: Weight decay coefficient, defaults to 1e-2
        :type weight_decay: float, optional
        :param amsgrad: Whether to use AMSGrad variant, defaults to False
        :type amsgrad: bool, optional
        :return: None
        :rtype: None
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad
        )
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        """Set the state of the optimizer.

        :param state: State dictionary to restore
        :type state: dict
        :return: None
        :rtype: None
        """
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)

    def step(self, closure=None):
        """Perform a single optimization step.

        :param closure: A closure that reevaluates the model and returns the loss, defaults to None
        :type closure: callable, optional
        :return: Loss value if closure is provided
        :rtype: float or None
        """
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform stepweight decay
                p.data.mul_(1 - group["lr"] * group["weight_decay"])

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                amsgrad = group["amsgrad"]

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(
                    grad, alpha=1 - beta1
                )  # TODO: check if this is correct
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
