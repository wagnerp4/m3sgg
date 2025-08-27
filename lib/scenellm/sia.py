"""
Spatial Information Aggregator (SIA) implementation for SceneLLM.
Credit to the authors of the original code: https://doi.org/10.1016/j.patcog.2025.111992.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: Compare different clustering methods
try:
    import dgl

    DGL_AVAILABLE = True
except (ImportError, FileNotFoundError, OSError) as e:
    DGL_AVAILABLE = False
    print(
        f"Warning: DGL not available ({type(e).__name__}: {str(e)[:100]}...). SIA will use simplified graph structure."
    )

try:
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print(
        "Warning: scipy not available. Hierarchical clustering will use k-means fallback."
    )


def build_hierarchical_graph(boxes):
    """
    Build hierarchical graph from bounding boxes using hierarchical clustering.
    Args:
        boxes: tensor of shape [N, 4] containing normalized bounding boxes
    Returns:
        DGL graph or simple edge list based on hierarchical clustering
    """
    if boxes.size(0) <= 1:
        # Single node or empty - return simple structure
        if DGL_AVAILABLE:
            g = dgl.graph(
                (torch.tensor([0]), torch.tensor([0])), num_nodes=max(1, boxes.size(0))
            )
            return g
        else:
            return {
                "edges_src": [0],
                "edges_dst": [0],
                "num_nodes": max(1, boxes.size(0)),
            }

    boxes_np = boxes.detach().cpu().numpy()
    centers = boxes_np[:, :2] + boxes_np[:, 2:] / 2  # [x_center, y_center]
    if centers.shape[0] == 1:
        # Single node - return simple structure
        if DGL_AVAILABLE:
            g = dgl.graph((torch.tensor([0]), torch.tensor([0])), num_nodes=1)
            return g
        else:
            return {"edges_src": [0], "edges_dst": [0], "num_nodes": 1}

    # Build edges based on available libraries
    edges_src = []
    edges_dst = []

    # TODO: Explore other clustering methods
    if SCIPY_AVAILABLE:
        # Use hierarchical clustering
        distances = pdist(centers, metric="euclidean")
        linkage_matrix = linkage(distances, method="ward")
        max_clusters = min(boxes.size(0), max(2, boxes.size(0) // 2))
        clusters = fcluster(linkage_matrix, max_clusters, criterion="maxclust")
        # Add edges within clusters
        for cluster_id in np.unique(clusters):
            cluster_nodes = np.where(clusters == cluster_id)[0]
            if len(cluster_nodes) > 1:
                # Fully connect nodes within cluster
                for i in cluster_nodes:
                    for j in cluster_nodes:
                        if i != j:
                            edges_src.append(i)
                            edges_dst.append(j)
        # Add nearest neighbor connections between clusters
        for i in range(len(centers)):
            # Find nearest neighbor from different cluster
            for j in range(len(centers)):
                if i != j and clusters[i] != clusters[j]:
                    # Add edge to nearest neighbor in different cluster
                    edges_src.append(i)
                    edges_dst.append(j)
                    break  # Only one inter-cluster edge per node
    else:
        # Fallback: simple distance-based connectivity
        distances = np.linalg.norm(centers[:, None] - centers[None, :], axis=2)
        # Connect each node to its k nearest neighbors
        k = min(3, boxes.size(0) - 1)
        for i in range(len(centers)):
            nearest = np.argsort(distances[i])[1 : k + 1]  # Skip self (index 0)
            for j in nearest:
                edges_src.append(i)
                edges_dst.append(j)
    # Ensure we have at least some edges (fully connected if needed)
    if len(edges_src) == 0:
        # Fallback: create a simple connected graph
        for i in range(boxes.size(0)):
            for j in range(i + 1, boxes.size(0)):
                edges_src.extend([i, j])
                edges_dst.extend([j, i])
    # Create DGL graph or return edge structure
    if DGL_AVAILABLE:
        return dgl.graph(
            (torch.tensor(edges_src), torch.tensor(edges_dst)), num_nodes=boxes.size(0)
        )
    else:
        return {
            "edges_src": edges_src,
            "edges_dst": edges_dst,
            "num_nodes": boxes.size(0),
        }


class SIA(nn.Module):
    def __init__(self, dim=1024):
        """
        Spatial Information Aggregator - Embed (x, y, w, h)
        then fuse ROI tokens with spatial reasoning.
        """
        super().__init__()
        self.dim = dim
        self.pos_mlp = nn.Sequential(nn.Linear(4, dim), nn.ReLU(), nn.Linear(dim, dim))
        if DGL_AVAILABLE:
            self.gcn1 = dgl.nn.GraphConv(dim, dim, allow_zero_in_degree=True)
            self.gcn2 = dgl.nn.GraphConv(dim, dim, allow_zero_in_degree=True)
        else:
            # Fallback: use simple linear layers with attention
            self.spatial_attn = nn.MultiheadAttention(
                dim, num_heads=8, batch_first=True
            )
            self.linear1 = nn.Linear(dim, dim)
            self.linear2 = nn.Linear(dim, dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, feats, boxes):  # feats/z_q [R, D], boxes [R, 4]
        if feats.size(0) == 0:  # Handle empty input
            return torch.zeros(self.dim, device=feats.device)

        pos = self.pos_mlp(boxes)  # [R, D]
        h = feats + pos  # [R, D]

        if DGL_AVAILABLE:
            g = build_hierarchical_graph(boxes)  # DGL graph
            g = dgl.add_self_loop(g)  # Add self-loops to handle 0-in-degree nodes
            g = g.to(feats.device)
            h = F.relu(self.gcn1(g, h))
            h = self.dropout(h)
            h = self.gcn2(g, h)
        else:
            # Fallback: use self-attention for spatial reasoning
            h_unsqueezed = h.unsqueeze(0)  # [1, R, D]
            h_attn, _ = self.spatial_attn(h_unsqueezed, h_unsqueezed, h_unsqueezed)
            h = h_attn.squeeze(0)  # [R, D]
            h = F.relu(self.linear1(h))
            h = self.dropout(h)
            h = self.linear2(h)

        # Aggregate to single frame token ("Chinese character" representation)
        frame_token = h.mean(0)  # [D]
        return frame_token  # [D]
