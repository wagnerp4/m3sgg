"""
VQ-VAE Quantizer implementation for SceneLLM.
Credit to the authors of the original code: https://doi.org/10.1016/j.patcog.2025.111992.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VQVAEQuantizer(nn.Module):
    """Vector Quantized Variational AutoEncoder (VQ-VAE) quantizer.
    
    Implements discrete latent space quantization for scene representations
    with codebook learning and commitment loss for stable training.
    
    :param nn.Module: Base PyTorch module class
    :type nn.Module: class
    """
    
    def __init__(
        self, input_dim=2048, dim=1024, codebook_size=8192, commitment_cost=0.25
    ):
        """Initialize the VQ-VAE quantizer.
        
        :param input_dim: Input feature dimension, defaults to 2048
        :type input_dim: int, optional
        :param dim: Latent dimension, defaults to 1024
        :type dim: int, optional
        :param codebook_size: Size of the discrete codebook, defaults to 8192
        :type codebook_size: int, optional
        :param commitment_cost: Weight for commitment loss, defaults to 0.25
        :type commitment_cost: float, optional
        :return: None
        :rtype: None
        """
        super().__init__()
        self.input_dim = input_dim
        self.dim = dim
        self.codebook_size = codebook_size
        self.commitment_cost = commitment_cost
        self.input_projection = nn.Linear(input_dim, dim)
        self.encoder = nn.Linear(dim, dim)
        self.codebook = nn.Embedding(codebook_size, dim)
        self.decoder = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, input_dim)
        self.codebook.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)
        # For tracking usage
        self.register_buffer("usage_count", torch.zeros(codebook_size))

    def forward(self, roi_feats):
        """Forward pass through VQ-VAE quantizer.
        
        :param roi_feats: ROI features tensor of shape [N, input_dim]
        :type roi_feats: torch.Tensor
        :return: Tuple containing reconstructed features, reconstruction loss, embedding loss, and commitment loss
        :rtype: tuple
        """
        z_e = self.encoder(self.input_projection(roi_feats))  # [N, dim]
        z_e_flat = z_e.view(-1, self.dim)  # [N, D]
        d = (
            z_e_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_e_flat @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(1)
        )  # [N, codebook_size]
        min_encoding_indices = d.argmin(-1)  # [N] Find nearest codebook entries
        z_q = self.codebook(min_encoding_indices)  # [N, D]
        if self.training:
            usage_one_hot = F.one_hot(min_encoding_indices, self.codebook_size).float()
            self.usage_count += usage_one_hot.sum(0)

        z_q = z_e + (z_q - z_e).detach()  # Straight through estimator

        # Reconstruct
        decoded = self.decoder(z_q)
        recon = self.output_projection(decoded)  # Project back to input dimension
        recon_loss = F.mse_loss(recon, roi_feats)  # Calculate VQ-VAE losses
        embedding_loss = F.mse_loss(
            z_q.detach(), z_e
        )  # Embedding loss: move embeddings towards encoder outputs
        commitment_loss = F.mse_loss(
            z_q, z_e.detach()
        )  # Commitment loss: encourage encoder to commit to embeddings
        vq_loss = recon_loss + embedding_loss + self.commitment_cost * commitment_loss
        return {
            "ids": min_encoding_indices,
            "z_q": z_q,
            "recon": recon,
            "vq_loss": vq_loss,
            "recon_loss": recon_loss,
            "embedding_loss": embedding_loss,
            "commitment_loss": commitment_loss,
        }

    def get_usage_histogram(self):
        """Get current usage histogram for OT update."""
        return self.usage_count.clone()

    def reset_usage_count(self):
        """Reset usage counter."""
        self.usage_count.zero_()

    def update_codebook(self, new_codebook_weights):
        """Update codebook with new weights."""
        self.codebook.weight.data = new_codebook_weights
        self.reset_usage_count()  # Reset usage count after update
