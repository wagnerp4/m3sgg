"""
Optimal Transport Codebook Updater implementation for SceneLLM.
Credit to the authors of the original code: https://doi.org/10.1016/j.patcog.2025.111992.
"""

import torch
import torch.nn.functional as F

try:
    import ot

    OT_AVAILABLE = True
except ImportError as e:
    OT_AVAILABLE = False
    print(f"Warning: POT not available ({e}). Install with 'pip install pot'. OT codebook update will use simple merging.")


class OTCodebookUpdater:
    def __init__(self, base_codebook, step=512, max_iterations=10):
        self.step = step
        self.C = base_codebook  # nn.Embedding
        self.max_iterations = max_iterations
        self.reg = 0.01  # Regularization parameter for Sinkhorn

    def update(self, usage_hist):
        """
        Update codebook using Optimal Transport scheme.
        Args:
            usage_hist: tensor of shape [codebook_size] with usage frequencies
        Returns:
            updated_codebook: new embedding weight matrix
        """
        codebook_size = self.C.weight.size(0)
        embed_dim = self.C.weight.size(1)
        usage_prob = F.softmax(
            usage_hist + 1e-8, dim=0
        )  # Add small epsilon to avoid zeros
        current_entropy = -torch.sum(usage_prob * torch.log(usage_prob + 1e-8))
        best_codebook = self.C.weight.clone()

        for k in range(1, self.max_iterations + 1):
            new_size = max(
                codebook_size - k * self.step, self.step
            )  # Reduce size gradually
            if new_size >= codebook_size:
                continue

            c = self._compute_cost_matrix(usage_prob, new_size)
            transport_plan = self._sinkhorn_solver(usage_prob, new_size, c)
            new_codebook = self._generate_new_codebook(
                transport_plan, new_size, embed_dim
            )

            # Calculate new entropy (simplified)
            new_usage = torch.sum(transport_plan, dim=0)
            new_usage_prob = F.softmax(new_usage + 1e-8, dim=0)
            new_entropy = -torch.sum(new_usage_prob * torch.log(new_usage_prob + 1e-8))

            # Check if entropy decreased significantly
            entropy_decrease = current_entropy - new_entropy
            if entropy_decrease > 0.01:  # Threshold for significant decrease
                best_codebook = new_codebook
                break

        return best_codebook

    def _compute_cost_matrix(self, usage_prob, new_size):
        """Compute cost matrix for transport.
        TODO: Improve cost matrix with semantic similarity.
        """
        device = usage_prob.device
        old_size = usage_prob.size(0)

        # Simple cost based on indices
        cost_matrix = torch.zeros(old_size, new_size, device=device)
        for i in range(old_size):
            for j in range(new_size):
                # Cost based on distance between indices (normalized)
                cost_matrix[i, j] = abs(i / old_size - j / new_size)

        return cost_matrix

    def _sinkhorn_solver(self, source_prob, target_size, cost_matrix):
        """Solve optimal transport using Sinkhorn algorithm."""
        device = source_prob.device
        source_size = source_prob.size(0)
        target_prob = torch.ones(target_size, device=device) / target_size

        # Convert to numpy for POT library
        source_np = source_prob.detach().cpu().numpy()
        target_np = target_prob.detach().cpu().numpy()
        cost_np = cost_matrix.detach().cpu().numpy()

        # Solve OT problem
        if OT_AVAILABLE:
            try:
                transport_plan_np = ot.sinkhorn(
                    source_np, target_np, cost_np, self.reg, numItermax=100
                )
                transport_plan = torch.from_numpy(transport_plan_np).to(device)
            except Exception:
                # Fallback: uniform transport plan
                transport_plan = torch.ones(source_size, target_size, device=device)
                transport_plan = transport_plan / transport_plan.sum(
                    dim=1, keepdim=True
                )
        else:
            # Simple fallback when OT library not available
            transport_plan = torch.ones(source_size, target_size, device=device)
            transport_plan = transport_plan / transport_plan.sum(dim=1, keepdim=True)

        return transport_plan

    def _generate_new_codebook(self, transport_plan, new_size, embed_dim):
        """Generate new codebook based on transport plan."""
        device = self.C.weight.device

        # Weighted combination of old codewords
        new_codebook = torch.zeros(new_size, embed_dim, device=device)

        for j in range(new_size):
            # Normalize weights for this new codeword
            weights = transport_plan[:, j]
            weights = weights / (weights.sum() + 1e-8)

            # Weighted average of old codewords
            new_codebook[j] = torch.sum(weights.unsqueeze(1) * self.C.weight, dim=0)

        return new_codebook
