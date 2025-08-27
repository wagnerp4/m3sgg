"""
Main SceneLLM network and SGG decoder implementation.
Credit to the authors of the original code: https://doi.org/10.1016/j.patcog.2025.111992.
"""

import torch
import torch.nn as nn
from lib.sttran import STTran
from .vqvae import VQVAEQuantizer
from .sia import SIA
from .ot import OTCodebookUpdater, OT_AVAILABLE
from .llm import SceneLLMLoRA

# TODO: Improve GCN architecture
# TODO: Use Cross Entropy instead of MSE


class SceneLLM(nn.Module):
    def __init__(self, cfg, dataset):
        super().__init__()
        self.cfg = cfg
        D = cfg.embed_dim
        # VQ-VAE
        self.quantiser = VQVAEQuantizer(
            input_dim=2048,
            dim=D,
            codebook_size=cfg.codebook_size,
            commitment_cost=cfg.commitment_cost,
        )
        # SIA
        self.sia = SIA(D)
        # OT
        if OT_AVAILABLE:
            self.ot_updater = OTCodebookUpdater(self.quantiser.codebook, cfg.ot_step)
        else:
            self.ot_updater = None
        # LLM
        self.llm = SceneLLMLoRA(
            cfg.llm_name,
            fallback_dim=D,
            r=cfg.lora_r,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
        )
        llm_hidden_size = self.llm.hidden_size
        self.llm_input_projection = (
            nn.Linear(D, llm_hidden_size) if llm_hidden_size != D else nn.Identity()
        )
        # SGG
        self.decoder = STTran(
            mode=cfg.mode,
            attention_class_num=len(dataset.attention_relationships),
            spatial_class_num=len(dataset.spatial_relationships),
            contact_class_num=len(dataset.contacting_relationships),
            obj_classes=dataset.object_classes,
            enc_layer_num=getattr(cfg, "enc_layer", 1),
            dec_layer_num=getattr(cfg, "dec_layer", 3),
        )

        # Projections
        self.feature_projection = nn.Linear(llm_hidden_size, 2048)
        self.roi_feature_projection = nn.Linear(D, 2048)
        self.fallback_projection = nn.Linear(D, 2048)

        # Training stage
        self.training_stage = "vqvae"  # 'vqvae', 'stage1', 'stage2'
        self.freeze_vqvae = False
        self.freeze_llm = True
        self.dataset = dataset

    def forward(self, entry):
        roi_feat = entry["features"]  # [R, D]
        boxes_tensor = entry["boxes"]

        if boxes_tensor.size(1) == 5:
            # STTran format: [batch_idx, x1, y1, x2, y2] -> extract [x1, y1, x2, y2]
            spatial_boxes = boxes_tensor[:, 1:]  # [R, 4]
        else:
            # Regular format: [x1, y1, x2, y2] or [x, y, w, h]
            spatial_boxes = boxes_tensor  # [R, 4]

        # Normalize boxes to [0,1]
        if "im_wh" in entry:
            im_wh = entry["im_wh"]  # [W, H]
            if len(im_wh.shape) == 1 and im_wh.size(0) == 2:
                # Expand to match box dimensions [x, y, w, h] or [x1, y1, x2, y2]
                norm_factors = torch.cat([im_wh, im_wh])  # [W, H, W, H]
            else:
                norm_factors = im_wh
            boxes = spatial_boxes / norm_factors  # normalize to [0,1]
        else:
            # Fallback: assume boxes are already normalized or use default image size
            # For Action Genome, typical image size is around 480x360, but we'll normalize by max coord
            max_coords = torch.max(spatial_boxes, dim=0)[0]  # [4]
            if torch.any(max_coords > 2.0):  # If coordinates seem to be in pixel space
                # Estimate normalization - use max values as rough image dimensions
                norm_factors = torch.cat(
                    [max_coords[:2], max_coords[:2]]
                )  # [x_max, y_max, x_max, y_max]
                boxes = spatial_boxes / norm_factors.clamp(
                    min=1.0
                )  # normalize to [0,1]
            else:
                # Assume already normalized
                boxes = spatial_boxes

        vq_results = self.quantiser(roi_feat)  # VQ-VAE
        code_vecs = vq_results["z_q"]  # [R, D]
        
        # Check for NaN in quantized vectors
        if torch.isnan(code_vecs).any():
            print("WARNING: NaN detected in VQ-VAE output, using input features directly")
            code_vecs = roi_feat  # Fallback to original features
            
        frame_tok = self.sia(code_vecs, boxes)  # [D] SIA
        
        # Check for NaN in SIA output
        if torch.isnan(frame_tok).any():
            print("WARNING: NaN detected in SIA output, using mean of input")
            frame_tok = code_vecs.mean(0)  # Use mean of code vectors as fallback

        # For video sequences, we need to handle temporal dimension
        # Currently handling single frame - extend for video sequences
        if frame_tok.dim() == 1:
            frame_tok = frame_tok.unsqueeze(0).unsqueeze(0)  # [1, 1, D]
        elif frame_tok.dim() == 2:
            frame_tok = frame_tok.unsqueeze(0)  # [1, T, D]

        # LLM reasoning (only if not frozen and not in VQ-VAE stage)
        if not self.freeze_llm and self.training_stage != "vqvae":
            # Project embeddings to match LLM input dimension
            projected_frame_tok = self.llm_input_projection(
                frame_tok
            )  # [B, T, D] -> [B, T, llm_hidden_size]
            
            # Check for NaN in input to LLM
            if torch.isnan(projected_frame_tok).any():
                print("WARNING: NaN detected in LLM input, using fallback")
                # Use fallback: direct projection without LLM
                enhanced_frame_token = self.fallback_projection(
                    frame_tok.squeeze(0).squeeze(0)
                )  # [D] -> [2048]
            else:
                hidden = self.llm(projected_frame_tok)  # [B, T, hidden_size]
                
                # Check for NaN in LLM output
                if torch.isnan(hidden).any():
                    print("WARNING: NaN detected in LLM output, using fallback")
                    # Use fallback: direct projection without LLM
                    enhanced_frame_token = self.fallback_projection(
                        frame_tok.squeeze(0).squeeze(0)
                    )  # [D] -> [2048]
                else:
                    enhanced_frame_token = self.feature_projection(
                        hidden.squeeze(0).squeeze(0)
                    )  # [hidden_size] -> [2048]
        else:
            enhanced_frame_token = self.fallback_projection(
                frame_tok.squeeze(0).squeeze(0)
            )  # [D] -> [2048]

        # Enhance ROI features with SceneLLM reasoning
        enhanced_roi_features = self.roi_feature_projection(
            code_vecs
        )  # [R, D] -> [R, 2048]
        num_rois = enhanced_roi_features.size(0)
        frame_features = enhanced_frame_token.unsqueeze(0).expand(
            num_rois, -1
        )  # [R, 2048]
        combined_features = enhanced_roi_features + frame_features  # [R, 2048]
        sttran_entry = entry.copy()  # Start with original entry
        sttran_entry["features"] = combined_features  # Enhanced 2048-dim features

        if "distribution" in sttran_entry:
            dist = sttran_entry["distribution"]
            if dist.size(1) == len(self.dataset.object_classes):
                # Remove background class (first column)
                sttran_entry["distribution"] = dist[:, 1:]  # [R, num_classes-1]

        # SGG prediction using STTran with enhanced features
        pred = self.decoder(sttran_entry)
        pred.update(
            {
                "vq_loss": vq_results["vq_loss"],
                "recon_loss": vq_results["recon_loss"],
                "embedding_loss": vq_results["embedding_loss"],
                "commitment_loss": vq_results["commitment_loss"],
            }
        )
        return pred

    def set_training_stage(self, stage):
        """Set training stage and freeze/unfreeze components accordingly."""
        print(f"DEBUG: set_training_stage called with: {stage}")
        self.training_stage = stage
        print(f"DEBUG: training_stage set to: {self.training_stage}")

        # Components: {VQ-VAE, SIA, LLM, Decoder, MLP, GCN, SGG}
        if stage == "vqvae":
            # Stage 0: Pre-train VQ-VAE only
            self.freeze_vqvae = False
            self.freeze_llm = True
            # Freeze SIA, LLM, Decoder, and projections
            for param in self.sia.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = False
            for param in self.llm.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = False
            for param in self.decoder.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = False
            for param in self.llm_input_projection.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = False
            for param in self.feature_projection.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = False
            for param in self.roi_feature_projection.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = False
            for param in self.fallback_projection.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = False
            # Keep MLP, GCN, SGG unfrozen

        elif stage == "stage1":
            # Stage 1: Train MLP, GCN, SGG (freeze VQ-VAE, LLM)
            self.freeze_vqvae = True
            self.freeze_llm = True
            # Freeze VQ-VAE
            for param in self.quantiser.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = False
            # Unfreeze SIA and feature projections
            for param in self.sia.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = True
            for param in self.feature_projection.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = True
            for param in self.roi_feature_projection.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = True
            for param in self.fallback_projection.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = True
            # Keep decoder frozen (use STTran ckpt)
            for param in self.decoder.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = False
            # Keep LLM frozen
            for param in self.llm.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = False

        elif stage == "stage2":
            # Stage 2: Fine-tune with LoRA (unfreeze LLM with LoRA)
            self.freeze_vqvae = True
            self.freeze_llm = False
            # Keep VQ-VAE frozen
            for param in self.quantiser.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = False
            # Keep SIA and feature projections trainable
            for param in self.sia.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = True
            for param in self.feature_projection.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = True
            for param in self.roi_feature_projection.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = True
            for param in self.fallback_projection.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = True
            # Keep decoder frozen (it's STTran - can be trained separately)
            for param in self.decoder.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = False
            # Unfreeze LLM (LoRA parameters will be trainable)
            for param in self.llm.parameters():
                if param.dtype.is_floating_point:
                    param.requires_grad = True

    def update_codebook_with_ot(self):
        """Update codebook using Optimal Transport scheme."""
        if self.training_stage != "vqvae" and self.ot_updater is not None:
            usage_hist = self.quantiser.get_usage_histogram()
            new_codebook = self.ot_updater.update(usage_hist)
            self.quantiser.update_codebook(new_codebook)
            print(
                f"Updated codebook from {self.quantiser.codebook_size} to {new_codebook.size(0)} entries"
            )
        elif self.ot_updater is None:
            print("OT library not available - skipping codebook update")


# Deprecated: Replace with SGG module
class SGGDecoder(nn.Module):
    def __init__(self, hidden_dim, attn_c, spat_c, cont_c):
        super().__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_dim, 8), 3
        )
        self.attn_head = nn.Linear(hidden_dim, attn_c)
        self.spat_head = nn.Linear(hidden_dim, spat_c)
        self.cont_head = nn.Linear(hidden_dim, cont_c)

    def forward(self, seq):  # [B,T,D]
        h = self.transformer(seq)  # temporal reasoning
        g = h.mean(1)  # video-level node
        return {
            "attention_distribution": self.attn_head(g),
            "spatial_distribution": self.spat_head(g),
            "contact_distribution": self.cont_head(g),
        }
