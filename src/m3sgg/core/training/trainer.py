"""
Trainer class for modularized training loop.

"""

import copy
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from m3sgg.core.config.config import Config
from m3sgg.core.evaluation.metrics import BasicSceneGraphEvaluator
from m3sgg.utils.checkpoint_utils import safe_save_checkpoint
from .loss_computation import LossComputation


class Trainer:
    """Main trainer class for scene graph generation models.
    
    This class encapsulates the training loop, epoch management, and step execution
    for various scene graph generation models.
    
    :param model: The model to train
    :type model: torch.nn.Module
    :param config: Configuration object containing training parameters
    :type config: Config
    :param dataloader_train: Training data loader
    :type dataloader_train: torch.utils.data.DataLoader
    :param dataloader_test: Test data loader
    :type dataloader_test: torch.utils.data.DataLoader
    :param optimizer: Optimizer for training
    :type optimizer: Optimizer
    :param scheduler: Learning rate scheduler
    :type scheduler: ReduceLROnPlateau
    :param logger: Logger instance
    :type logger: logging.Logger
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Config,
        dataloader_train: torch.utils.data.DataLoader,
        dataloader_test: torch.utils.data.DataLoader,
        optimizer: Optimizer,
        scheduler: ReduceLROnPlateau,
        logger: logging.Logger,
        object_detector: Optional[torch.nn.Module] = None,
        object_detector_EASG: Optional[torch.nn.Module] = None,
        matcher: Optional[Any] = None,
        evaluator: Optional[BasicSceneGraphEvaluator] = None,
        evaluator2: Optional[BasicSceneGraphEvaluator] = None,
        dataset_train: Optional[Any] = None,
        dataset_test: Optional[Any] = None,
    ):
        """Initialize the Trainer with all necessary components.
        
        :param model: The model to train
        :param config: Configuration object
        :param dataloader_train: Training data loader
        :param dataloader_test: Test data loader
        :param optimizer: Optimizer
        :param scheduler: Learning rate scheduler
        :param logger: Logger instance
        :param object_detector: Object detector for Action Genome dataset
        :param object_detector_EASG: Object detector for EASG dataset
        :param matcher: Hungarian matcher for DSG-DETR
        :param evaluator: Scene graph evaluator
        :param evaluator2: Secondary evaluator without constraints
        :param dataset_train: Training dataset
        :param dataset_test: Test dataset
        """
        self.model = model
        self.config = config
        self.dataloader_train = dataloader_train
        self.dataloader_test = dataloader_test
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.object_detector = object_detector
        self.object_detector_EASG = object_detector_EASG
        self.matcher = matcher
        self.evaluator = evaluator
        self.evaluator2 = evaluator2
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        
        # Training state
        self.best_score = 0.0
        self.best_Mrecall = 0.0
        self.best_epoch = -1
        self.tr = []  # Training loss history
        
        # Loss functions (will be set by specific model types)
        self.ce_loss = None
        self.bce_loss = None
        self.mlm_loss = None
        self.ce_loss_obj = None
        self.ce_loss_rel = None
        self.con_loss = None
        
        # Loss computation helper
        self.loss_computer = None

    def _initialize_loss_functions(self) -> None:
        """Initialize loss functions based on configuration.
        
        This method sets up the appropriate loss functions based on the model type
        and configuration settings.
        """
        # Basic loss functions
        if self.config.bce_loss:
            self.ce_loss = nn.CrossEntropyLoss()
            self.bce_loss = nn.BCELoss()
        else:
            self.ce_loss = nn.CrossEntropyLoss()
            self.mlm_loss = nn.MultiLabelMarginLoss()

        # Model-specific loss functions
        if self.config.model_type == "tempura":
            weights = torch.ones(len(self.model.obj_classes))
            weights[0] = self.config.eos_coef
            if self.config.obj_head != "gmm":
                self.ce_loss_obj = nn.CrossEntropyLoss(
                    weight=weights.to(device=torch.device(self.config.device)), 
                    reduction="none"
                )
            else:
                self.ce_loss_obj = nn.NLLLoss(
                    weight=weights.to(device=torch.device(self.config.device)), 
                    reduction="none"
                )
            
            if self.config.rel_head != "gmm":
                self.ce_loss_rel = nn.CrossEntropyLoss(reduction="none")
            else:
                self.ce_loss_rel = nn.NLLLoss(reduction="none")
            
            if self.config.mlm:
                self.mlm_loss = nn.MultiLabelMarginLoss(reduction="none")
            else:
                self.bce_loss = nn.BCELoss(reduction="none")
            
            if self.config.obj_con_loss == "euc_con":
                from m3sgg.utils.infoNCE import EucNormLoss
                self.con_loss = EucNormLoss()
                self.con_loss.train()
            elif self.config.obj_con_loss == "info_nce":
                from m3sgg.utils.infoNCE import SupConLoss
                self.con_loss = SupConLoss(temperature=0.1)
                self.con_loss.train()

        # Initialize loss computation helper
        self.loss_computer = LossComputation(
            config=self.config,
            dataset_train=self.dataset_train,
            ce_loss=self.ce_loss,
            bce_loss=self.bce_loss,
            mlm_loss=self.mlm_loss,
            ce_loss_obj=self.ce_loss_obj,
            ce_loss_rel=self.ce_loss_rel,
            con_loss=self.con_loss,
        )

    def train_loop(self) -> None:
        """Main training loop that orchestrates the entire training process.
        
        This method runs the complete training process including all epochs,
        evaluation, and checkpoint saving.
        """
        # Initialize loss functions
        self._initialize_loss_functions()
        
        self.logger.info(
            "Starting training | optimizer=%s, lr=%e | train_batches=%d, test_batches=%d",
            self.config.optimizer,
            self.config.lr,
            len(self.dataloader_train),
            len(self.dataloader_test),
        )

        # Initialize predictions data collection
        all_predictions_data = []
        
        # Epoch loop
        for epoch in range(int(self.config.nepoch)):
            self.logger.info("=" * 40)
            self.logger.info("Starting Epoch %d", epoch)
            self.logger.info("=" * 40)

            # Train one epoch
            self.train_epoch(epoch)
            
            # Evaluate the model
            score, mrecall, predictions_data = self.evaluate_epoch(epoch)
            
            # Collect predictions data for CSV saving
            all_predictions_data.extend(predictions_data)
            
            # Update learning rate
            self.scheduler.step(score)
            
            # Save checkpoints if this is the best model
            self.save_checkpoints(epoch, score, mrecall)

        # Save final predictions as CSV (only for the best epoch)
        self.save_predictions_csv(all_predictions_data)
        
        self.logger.info("Training completed!")
        if not self.config.disable_checkpoint_saving:
            self.logger.info(
                "Best model saved at epoch %d with R@20 score: %.4f", 
                self.best_epoch, 
                self.best_score
            )
        else:
            self.logger.info(
                "Best model achieved at epoch %d with R@20 score: %.4f (checkpoint saving disabled)",
                self.best_epoch,
                self.best_score,
            )

    def train_epoch(self, epoch: int) -> None:
        """Train the model for one epoch.
        
        :param epoch: Current epoch number
        :type epoch: int
        """
        # Initialize uncertainty values for Tempura model
        if self.config.model_type == "tempura":
            from m3sgg.utils.uncertainty import uncertainty_values
            unc_vals = uncertainty_values(
                obj_classes=len(self.model.obj_classes),
                attention_class_num=self.model.attention_class_num,
                spatial_class_num=self.model.spatial_class_num,
                contact_class_num=self.model.contact_class_num,
            )
        else:
            unc_vals = None

        # Set model and detectors to training mode
        self.model.train()
        if self.config.dataset == "action_genome":
            self.object_detector.is_train = True
            self.object_detector.train_x = True
        elif self.config.dataset == "EASG":
            self.object_detector_EASG.is_train = True
        else:
            raise ValueError(f"Dataset '{self.config.dataset}' not supported")

        # Initialize epoch tracking
        start = time.time()
        train_iter = iter(self.dataloader_train)
        epoch_train_losses = []
        epoch_train_loss_components = []

        # Batch loop
        for b in tqdm(
            range(len(self.dataloader_train)), 
            desc=f"Epoch {epoch}/{self.config.nepoch} [Train]"
        ):
            # Train one step
            losses = self.train_step(b, train_iter, unc_vals)
            
            # Track losses
            self.tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
            epoch_train_losses.append(sum(losses.values()).item())
            epoch_train_loss_components.append({x: y.item() for x, y in losses.items()})

            # Log progress every 1000 batches
            if b % 1000 == 0 and b >= 1000:
                time_per_batch = (time.time() - start) / 1000
                self.logger.info(
                    "e%d  b%d/%d  %.3fs/batch, %.1fm/epoch",
                    epoch,
                    b,
                    len(self.dataloader_train),
                    time_per_batch,
                    len(self.dataloader_train) * time_per_batch / 60,
                )
                mn = pd.concat(self.tr[-1000:], axis=1).mean(1)
                self.logger.info("Loss stats - %s", mn.to_dict())
                start = time.time()

        # Log epoch summary
        avg_train_loss = np.mean(epoch_train_losses)
        avg_train_loss_components = {
            k: round(
                float(np.mean([d.get(k, 0.0) for d in epoch_train_loss_components])), 4
            )
            for k in epoch_train_loss_components[0].keys()
        }
        self.logger.info(
            "Epoch %d | avg_train_loss=%.6f | components=%s | evaluating",
            epoch,
            avg_train_loss,
            avg_train_loss_components,
        )

    def train_step(
        self, 
        batch_idx: int, 
        train_iter: iter, 
        unc_vals: Optional[Any] = None
    ) -> Dict[str, torch.Tensor]:
        """Execute one training step.
        
        :param batch_idx: Current batch index
        :type batch_idx: int
        :param train_iter: Training data iterator
        :type train_iter: iter
        :param unc_vals: Uncertainty values for Tempura model
        :type unc_vals: Optional[Any]
        :return: Dictionary of loss components
        :rtype: Dict[str, torch.Tensor]
        """
        # Get batch data
        data = next(train_iter)
        im_data = copy.deepcopy(data[0].cuda(0))
        im_info = copy.deepcopy(data[1].cuda(0))

        # Process data through object detector
        entry = self._process_batch_data(data, im_data, im_info)

        # Apply matcher if needed
        if self.config.use_matcher and self.config.dataset != "EASG":
            from m3sgg.utils.track import get_sequence
            gt_annotation = self.dataset_train.gt_annotations[data[4]]
            get_sequence(
                entry,
                gt_annotation,
                self.matcher,
                (im_info[0][:2] / im_info[0, 2]).cpu().data,
                self.config.mode,
            )

        # Forward pass through model
        if self.config.model_type == "vlm":
            pred = self.model(entry, im_data)
        else:
            pred = self.model(entry)

        # Model-specific post-processing
        self._post_process_predictions(pred, batch_idx, data, unc_vals)

        # Compute losses
        losses = self._compute_losses(pred, data, unc_vals)

        # Backward pass
        self.optimizer.zero_grad()
        loss = sum(losses.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5, norm_type=2)
        self.optimizer.step()

        return losses

    def train_iter(self, max_iterations: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Iterative training function that yields training progress.
        
        This function provides an iterator interface for training, allowing for
        more granular control over the training process and real-time monitoring.
        
        :param max_iterations: Maximum number of iterations to run, defaults to None (run until epoch complete)
        :type max_iterations: Optional[int]
        :yield: Dictionary containing training progress information
        :rtype: Iterator[Dict[str, Any]]
        """
        # Initialize loss functions if not already done
        if self.loss_computer is None:
            self._initialize_loss_functions()
        
        # Initialize uncertainty values for Tempura model
        if self.config.model_type == "tempura":
            from m3sgg.utils.uncertainty import uncertainty_values
            unc_vals = uncertainty_values(
                obj_classes=len(self.model.obj_classes),
                attention_class_num=self.model.attention_class_num,
                spatial_class_num=self.model.spatial_class_num,
                contact_class_num=self.model.contact_class_num,
            )
        else:
            unc_vals = None

        # Set model and detectors to training mode
        self.model.train()
        if self.config.dataset == "action_genome":
            self.object_detector.is_train = True
            self.object_detector.train_x = True
        elif self.config.dataset == "EASG":
            self.object_detector_EASG.is_train = True
        else:
            raise ValueError(f"Dataset '{self.config.dataset}' not supported")

        # Initialize iteration tracking
        train_iter = iter(self.dataloader_train)
        iteration = 0
        epoch_losses = []
        epoch_loss_components = []
        
        # Determine total iterations
        total_iterations = min(len(self.dataloader_train), max_iterations) if max_iterations else len(self.dataloader_train)
        
        self.logger.info(f"Starting iterative training for {total_iterations} iterations")
        
        while iteration < total_iterations:
            try:
                # Execute one training step
                losses = self.train_step(iteration, train_iter, unc_vals)
                
                # Track losses
                self.tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
                epoch_losses.append(sum(losses.values()).item())
                epoch_loss_components.append({x: y.item() for x, y in losses.items()})
                
                # Prepare progress information
                progress_info = {
                    "iteration": iteration,
                    "total_iterations": total_iterations,
                    "losses": {x: y.item() for x, y in losses.items()},
                    "total_loss": sum(losses.values()).item(),
                    "progress": (iteration + 1) / total_iterations,
                }
                
                # Add recent loss statistics if available
                if len(self.tr) >= 100:
                    recent_losses = pd.concat(self.tr[-100:], axis=1).mean(1)
                    progress_info["recent_loss_stats"] = recent_losses.to_dict()
                
                yield progress_info
                iteration += 1
                
            except StopIteration:
                # End of epoch reached
                break
        
        # Calculate epoch summary
        avg_epoch_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_epoch_loss_components = {
            k: round(
                float(np.mean([d.get(k, 0.0) for d in epoch_loss_components])), 4
            )
            for k in epoch_loss_components[0].keys() if epoch_loss_components
        }
        
        # Yield final epoch summary
        yield {
            "iteration": iteration,
            "total_iterations": total_iterations,
            "epoch_complete": True,
            "avg_epoch_loss": avg_epoch_loss,
            "avg_epoch_loss_components": avg_epoch_loss_components,
            "progress": 1.0,
        }

    def _process_batch_data(self, data: tuple, im_data: torch.Tensor, im_info: torch.Tensor) -> Dict[str, Any]:
        """Process batch data through the appropriate object detector.
        
        :param data: Raw batch data
        :type data: tuple
        :param im_data: Image data tensor
        :type im_data: torch.Tensor
        :param im_info: Image info tensor
        :type im_info: torch.Tensor
        :return: Processed entry dictionary
        :rtype: Dict[str, Any]
        """
        if self.config.dataset == "EASG":
            gt_grounding = self.dataset_train.gt_groundings[data[2]]
            with torch.no_grad():
                entry = self.object_detector_EASG(
                    im_data, im_info, gt_grounding, im_all=None
                )
            entry["features_verb"] = copy.deepcopy(
                self.dataset_train.verb_feats[data[2]].cuda(0)
            )
        else:
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = self.dataset_train.gt_annotations[data[4]]
            with torch.no_grad():
                entry = self.object_detector(
                    im_data,
                    im_info,
                    gt_boxes,
                    num_boxes,
                    gt_annotation,
                    im_all=None,
                )
        return entry

    def _post_process_predictions(
        self, 
        pred: Dict[str, Any], 
        batch_idx: int, 
        data: tuple, 
        unc_vals: Optional[Any]
    ) -> None:
        """Apply model-specific post-processing to predictions.
        
        :param pred: Model predictions
        :type pred: Dict[str, Any]
        :param batch_idx: Current batch index
        :type batch_idx: int
        :param data: Raw batch data
        :type data: tuple
        :param unc_vals: Uncertainty values for Tempura model
        :type unc_vals: Optional[Any]
        """
        # SceneLLM: Periodic codebook update
        if self.config.model_type == "scenellm":
            if batch_idx > 0 and batch_idx % 1000 == 0 and self.config.scenellm_training_stage != "vqvae":
                self.model.update_codebook_with_ot()

        # Tempura: Uncertainty computation
        if (
            self.config.obj_unc
            or self.config.rel_unc
            or self.config.obj_mem_compute
            or self.config.rel_mem_compute
        ):
            from m3sgg.utils.uncertainty import uncertainty_computation
            uncertainty_computation(
                data,
                self.dataset_train,
                self.object_detector,
                self.model,
                unc_vals,
                torch.device(self.config.device),
                self.config.save_path,
                obj_unc=self.config.obj_unc,
                obj_mem=self.config.obj_mem_compute,
                background_mem=False,
                rel_unc=self.config.rel_unc,
            )

    def _compute_losses(self, pred: Dict[str, Any], data: tuple, unc_vals: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        """Compute losses based on model type and dataset.
        
        :param pred: Model predictions
        :type pred: Dict[str, Any]
        :param data: Raw batch data
        :type data: tuple
        :return: Dictionary of loss components
        :rtype: Dict[str, torch.Tensor]
        """
        # This is a simplified version - the full implementation would include
        # all the complex loss computation logic from the original train.py
        # For now, we'll delegate to a separate method for each model type
        
        if self.config.dataset == "EASG":
            return self._compute_easg_losses(pred)
        elif self.config.dataset == "action_genome":
            return self._compute_action_genome_losses(pred, unc_vals)
        else:
            raise ValueError(f"Dataset '{self.config.dataset}' not supported")

    def _compute_easg_losses(self, pred: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute losses for EASG dataset.
        
        :param pred: Model predictions
        :type pred: Dict[str, Any]
        :return: Dictionary of loss components
        :rtype: Dict[str, torch.Tensor]
        """
        edge_distribution = pred["edge_distribution"]
        losses = {}
        
        if self.config.mode != "edgecls":
            losses["obj_loss"] = self.ce_loss(pred["distribution"], pred["labels"])
        
        if self.config.mode == "easgcls":
            losses["verb_loss"] = self.ce_loss(
                pred["distribution_verb"], pred["labels_verb"]
            )
        
        edge_label = -torch.ones(
            [len(pred["edge"]), len(self.dataset_train.edge_classes)],
            dtype=torch.long,
        ).to(device=edge_distribution.device)
        
        for i in range(len(pred["edge"])):
            edge_label[i, : len(pred["edge"][i])] = torch.tensor(
                pred["edge"][i]
            )
        
        losses["edge_loss"] = self.mlm_loss(edge_distribution, edge_label)
        return losses

    def _compute_action_genome_losses(self, pred: Dict[str, Any], unc_vals: Optional[Any] = None) -> Dict[str, torch.Tensor]:
        """Compute losses for Action Genome dataset.
        
        :param pred: Model predictions
        :type pred: Dict[str, Any]
        :return: Dictionary of loss components
        :rtype: Dict[str, torch.Tensor]
        """
        if self.config.model_type in ["sttran", "dsg-detr", "vlm"]:
            return self._compute_sttran_losses(pred)
        elif self.config.model_type == "stket":
            return self.loss_computer.compute_stket_losses(pred)
        elif self.config.model_type == "tempura":
            return self.loss_computer.compute_tempura_losses(pred, unc_vals)
        elif self.config.model_type == "scenellm":
            return self.loss_computer.compute_scenellm_losses(pred)
        elif self.config.model_type == "oed":
            return self.loss_computer.compute_oed_losses(pred, self.model)
        else:
            raise ValueError(f"Model type '{self.config.model_type}' not supported")

    def _compute_sttran_losses(self, pred: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Compute losses for STTran, DSG-DETR, and VLM models.
        
        :param pred: Model predictions
        :type pred: Dict[str, Any]
        :return: Dictionary of loss components
        :rtype: Dict[str, torch.Tensor]
        """
        attention_distribution = pred["attention_distribution"]
        spatial_distribution = pred["spatial_distribution"]
        contact_distribution = pred["contact_distribution"]
        
        attention_label = (
            torch.tensor(pred["attention_gt"], dtype=torch.long)
            .to(device=attention_distribution.device)
            .squeeze()
        )
        
        # Ensure attention_label is 1D for CrossEntropyLoss
        if attention_label.dim() > 1:
            attention_label = attention_label.flatten()
        
        if not self.config.bce_loss:
            # multi-label margin loss or adaptive loss
            spatial_label = -torch.ones(
                [len(pred["spatial_gt"]), 6], dtype=torch.long
            ).to(device=attention_distribution.device)
            contact_label = -torch.ones(
                [len(pred["contact_gt"]), 17], dtype=torch.long
            ).to(device=attention_distribution.device)
            
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, : len(pred["spatial_gt"][i])] = (
                    torch.tensor(pred["spatial_gt"][i])
                )
                contact_label[i, : len(pred["contact_gt"][i])] = (
                    torch.tensor(pred["contact_gt"][i])
                )
        else:
            # bce loss
            spatial_label = torch.zeros(
                [len(pred["spatial_gt"]), 6], dtype=torch.float32
            ).to(device=attention_distribution.device)
            contact_label = torch.zeros(
                [len(pred["contact_gt"]), 17], dtype=torch.float32
            ).to(device=attention_distribution.device)
            
            for i in range(len(pred["spatial_gt"])):
                spatial_label[i, pred["spatial_gt"][i]] = 1
                contact_label[i, pred["contact_gt"][i]] = 1

        losses = {}
        if self.config.mode == "sgcls" or self.config.mode == "sgdet":
            losses["object_loss"] = self.ce_loss(
                pred["distribution"], pred["labels"]
            )

        losses["attention_relation_loss"] = self.ce_loss(
            attention_distribution, attention_label
        )
        
        if not self.config.bce_loss:
            losses["spatial_relation_loss"] = self.mlm_loss(
                spatial_distribution, spatial_label
            )
            losses["contact_relation_loss"] = self.mlm_loss(
                contact_distribution, contact_label
            )
        else:
            losses["spatial_relation_loss"] = self.bce_loss(
                spatial_distribution, spatial_label
            )
            losses["contact_relation_loss"] = self.bce_loss(
                contact_distribution, contact_label
            )
        
        return losses

    def evaluate_epoch(self, epoch: int) -> Tuple[float, float]:
        """Evaluate the model for one epoch.
        
        :param epoch: Current epoch number
        :type epoch: int
        :return: Tuple of (score, mrecall)
        :rtype: Tuple[float, float]
        """
        # Set model to evaluation mode
        self.model.eval()
        if self.config.dataset == "EASG":
            self.object_detector_EASG.is_train = False
        else:
            self.object_detector.is_train = False

        # Initialize evaluation tracking
        val_losses = []
        val_loss_components = []
        predictions_data = []

        # EASG-specific evaluation setup
        if self.config.dataset == "EASG":
            list_k = [10, 20, 50]
            recall_with = {k: [] for k in list_k}
            recall_no = {k: [] for k in list_k}

        # Evaluation loop
        with torch.no_grad():
            test_iter = iter(self.dataloader_test)
            for b in tqdm(
                range(len(self.dataloader_test)), 
                desc=f"Epoch {epoch}/{self.config.nepoch} [Eval]"
            ):
                data = next(test_iter)
                im_data = copy.deepcopy(data[0].cuda(0))
                im_info = copy.deepcopy(data[1].cuda(0))

                # Process validation data
                if self.config.dataset == "EASG":
                    # EASG evaluation logic
                    score, mrecall = self._evaluate_easg_batch(
                        data, im_data, im_info, recall_with, recall_no, 
                        predictions_data, epoch, val_losses, val_loss_components
                    )
                elif self.config.dataset == "action_genome":
                    # Action Genome evaluation logic
                    score, mrecall = self._evaluate_action_genome_batch(
                        data, im_data, im_info, predictions_data, epoch, 
                        val_losses, val_loss_components
                    )
                else:
                    raise ValueError(f"Dataset '{self.config.dataset}' not supported")

        # Log validation results
        avg_val_loss = np.mean(val_losses) if val_losses else 0.0
        if val_loss_components:
            avg_val_loss_components = {
                k: round(
                    float(np.mean([d.get(k, 0.0) for d in val_loss_components])), 4
                )
                for k in val_loss_components[0].keys()
            }
        else:
            avg_val_loss_components = {}
        self.logger.info(
            "Epoch %d | avg_val_loss=%.6f | components=%s",
            epoch,
            avg_val_loss,
            avg_val_loss_components,
        )

        # Compute final scores
        if self.config.dataset == "EASG":
            for k in list_k:
                recall_with[k] = sum(recall_with[k]) / len(recall_with[k]) * 100
                recall_no[k] = sum(recall_no[k]) / len(recall_no[k]) * 100
            score = (recall_with[20] + recall_no[20]) / 2
            mrecall = 0.0  # EASG doesn't use mrecall
        elif self.config.dataset == "action_genome":
            if len(self.evaluator.result_dict[self.config.mode + "_recall"][20]) == 0:
                score = 0.0
                mrecall = 0.0
                self.logger.warning("No evaluation results found - using default scores")
            else:
                score = np.mean(self.evaluator.result_dict[self.config.mode + "_recall"][20])
                mrecall = self.evaluator.calc_mrecall()[20]
            self.evaluator.print_stats()
            self.evaluator.reset_result()

        return score, mrecall, predictions_data

    def _evaluate_easg_batch(
        self, 
        data: tuple, 
        im_data: torch.Tensor, 
        im_info: torch.Tensor,
        recall_with: Dict[int, List[float]], 
        recall_no: Dict[int, List[float]],
        predictions_data: List[Dict], 
        epoch: int,
        val_losses: List[float], 
        val_loss_components: List[Dict]
    ) -> Tuple[float, float]:
        """Evaluate one EASG batch.
        
        :param data: Batch data
        :param im_data: Image data
        :param im_info: Image info
        :param recall_with: Recall with constraints tracking
        :param recall_no: Recall without constraints tracking
        :param predictions_data: Predictions data list
        :param epoch: Current epoch
        :param val_losses: Validation losses list
        :param val_loss_components: Validation loss components list
        :return: Tuple of (score, mrecall)
        :rtype: Tuple[float, float]
        """
        # TODO: Implement EASG batch evaluation
        # This would contain the EASG-specific evaluation logic from the original code
        return 0.0, 0.0

    def _evaluate_action_genome_batch(
        self, 
        data: tuple, 
        im_data: torch.Tensor, 
        im_info: torch.Tensor,
        predictions_data: List[Dict], 
        epoch: int,
        val_losses: List[float], 
        val_loss_components: List[Dict]
    ) -> Tuple[float, float]:
        """Evaluate one Action Genome batch.
        
        :param data: Batch data
        :param im_data: Image data
        :param im_info: Image info
        :param predictions_data: Predictions data list
        :param epoch: Current epoch
        :param val_losses: Validation losses list
        :param val_loss_components: Validation loss components list
        :return: Tuple of (score, mrecall)
        :rtype: Tuple[float, float]
        """
        gt_boxes = copy.deepcopy(data[2].cuda(0))
        num_boxes = copy.deepcopy(data[3].cuda(0))
        gt_annotation = self.dataset_test.gt_annotations[data[4]]

        entry = self.object_detector(
            im_data,
            im_info,
            gt_boxes,
            num_boxes,
            gt_annotation,
            im_all=None,
        )

        if self.config.use_matcher:
            from m3sgg.utils.track import get_sequence
            get_sequence(
                entry,
                gt_annotation,
                self.matcher,
                (im_info[0][:2] / im_info[0, 2]).cpu().data,
                self.config.mode,
            )
        
        # Pass image data to VLM model if it's a VLM model
        if self.config.model_type == "vlm":
            pred = self.model(entry, im_data)
        else:
            pred = self.model(entry)

        # Only compute validation loss if ground truth annotations are available
        if (
            "attention_gt" in pred
            and "spatial_gt" in pred
            and "contact_gt" in pred
        ):
            # Compute validation loss using the same logic as training
            losses = self._compute_losses(pred, data)
            val_loss = sum(losses.values())
            val_losses.append(val_loss.item())
            val_loss_components.append({x: y.item() for x, y in losses.items()})

        # Add evaluator call for Action Genome dataset
        # Fix missing pred_scores field for TEMPURA and SceneLLM models
        if (
            self.config.model_type in ["tempura", "scenellm"]
            and "pred_scores" not in pred
        ):
            if "distribution" in pred:
                # Use the same logic as original TEMPURA: max over distribution[:, 1:]
                if pred["distribution"].shape[1] > 1:
                    pred["pred_scores"] = torch.max(
                        pred["distribution"][:, 1:], dim=1
                    )[0]
                else:
                    pred["pred_scores"] = torch.max(
                        pred["distribution"], dim=1
                    )[0]
            elif "labels" in pred:
                pred["pred_scores"] = torch.ones(
                    pred["labels"].shape[0], device=pred["labels"].device
                )
            else:
                self.logger.warning(
                    f"No distribution or labels available to create pred_scores for {self.config.model_type}"
                )

        # Skip evaluation for OED models since they're relation prediction models, not object detection models
        if self.config.model_type == "oed" and pred.get("skip_evaluation", False):
            self.logger.info(
                "Skipping evaluation for OED model (relation prediction only)"
            )
            return 0.0, 0.0

        # This is the key call that was missing!
        self.evaluator.evaluate_scene_graph(gt_annotation, pred)

        # Collect Action Genome predictions with per-sample metrics (only for the best epoch)
        if epoch == self.best_epoch or epoch == int(self.config.nepoch) - 1:
            # Fix missing pred_scores field for TEMPURA and SceneLLM models (for per-sample evaluation)
            if (
                self.config.model_type in ["tempura", "scenellm"]
                and "pred_scores" not in pred
            ):
                if "distribution" in pred:
                    # Use the same logic as original TEMPURA: max over distribution[:, 1:]
                    if pred["distribution"].shape[1] > 1:
                        pred["pred_scores"] = torch.max(
                            pred["distribution"][:, 1:], dim=1
                        )[0]
                    else:
                        pred["pred_scores"] = torch.max(
                            pred["distribution"], dim=1
                        )[0]
                elif "labels" in pred:
                    pred["pred_scores"] = torch.ones(
                        pred["labels"].shape[0],
                        device=pred["labels"].device,
                    )
                else:
                    self.logger.warning(
                        f"No distribution or labels available to create pred_scores for {self.config.model_type} per-sample evaluation"
                    )

            # Get per-sample metrics for this sample
            per_sample_metrics = self.evaluator.evaluate_scene_graph(
                gt_annotation, pred, return_per_sample=True
            )

            # Create prediction entry with per-sample metrics
            pred_entry = {
                "dataset": self.config.dataset,
                "model": self.config.model_type,
                "mode": self.config.mode,
                "annotation_id": data[4],
                "epoch": epoch,
                "best_score": self.best_score,
                "best_epoch": self.best_epoch,
            }

            # Add per-sample metrics if available
            if per_sample_metrics and len(per_sample_metrics) > 0:
                sample_metrics = per_sample_metrics[0]  # Take first frame's metrics
                pred_entry.update(
                    {
                        "r10": sample_metrics.get("r10", 0.0),
                        "r20": sample_metrics.get("r20", 0.0),
                        "r50": sample_metrics.get("r50", 0.0),
                        "r100": sample_metrics.get("r100", 0.0),
                        "mrecall": sample_metrics.get("mrecall", 0.0),
                    }
                )
            else:
                # Fallback to default values if no metrics available
                pred_entry.update(
                    {
                        "r10": 0.0,
                        "r20": 0.0,
                        "r50": 0.0,
                        "r100": 0.0,
                        "mrecall": 0.0,
                    }
                )

            predictions_data.append(pred_entry)

        return 0.0, 0.0  # Will be computed from evaluator results

    def save_predictions_csv(self, predictions_data: List[Dict]) -> None:
        """Save final predictions as CSV file.
        
        :param predictions_data: List of prediction dictionaries
        :type predictions_data: List[Dict]
        """
        if predictions_data:
            predictions_df = pd.DataFrame(predictions_data)
            predictions_csv_path = os.path.join(self.config.save_path, "predictions.csv")
            predictions_df.to_csv(predictions_csv_path, index=False)
            self.logger.info(f"Predictions saved to: {predictions_csv_path}")
        else:
            self.logger.warning("No predictions data collected to save")

    def save_checkpoints(self, epoch: int, score: float, mrecall: float) -> None:
        """Save model checkpoints if this is the best model.
        
        :param epoch: Current epoch number
        :type epoch: int
        :param score: Current score
        :type score: float
        :param mrecall: Current mrecall
        :type mrecall: float
        """
        # Save best model checkpoint
        if score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            
            # Handle Tempura memory saving
            if self.config.model_type == "tempura":
                if epoch > 0 and self.config.rel_mem_compute is not None:
                    if len(self.model.object_classifier.obj_memory) == 0:
                        pass  # object_memory = []
                    else:
                        pass  # object_memory = self.model.object_classifier.obj_memory.to("cpu")
                    rel_memory = self.model.rel_memory
                    if len(rel_memory) != 0:
                        rel_memory = {
                            k: rel_memory[k].to("cpu") for k in rel_memory.keys()
                        }
                else:
                    pass  # object_memory = []
                    rel_memory = []
            
            if not self.config.disable_checkpoint_saving:
                checkpoint_saved = safe_save_checkpoint(
                    self.model,
                    os.path.join(self.config.save_path, "model_best.tar"),
                    self.config.model_type,
                    self.config.dataset,
                    additional_metadata={
                        "epoch": epoch,
                        "best_score": self.best_score,
                        "mode": self.config.mode,
                        "enc_layer": self.config.enc_layer,
                        "dec_layer": self.config.dec_layer,
                    },
                    logger=self.logger
                )
                if checkpoint_saved:
                    self.logger.info("NEW BEST! Saved best checkpoint after %d epochs", epoch)
                else:
                    self.logger.error("Failed to save best checkpoint - disabling future saves")
                    self.config.disable_checkpoint_saving = True
            else:
                self.logger.info("NEW BEST! Checkpoint saving disabled (epoch %d)", epoch)

        # Save best mrecall checkpoint
        if mrecall > self.best_Mrecall:
            self.best_Mrecall = mrecall
            if not self.config.disable_checkpoint_saving:
                checkpoint_saved = safe_save_checkpoint(
                    self.model,
                    os.path.join(self.config.save_path, "model_best_Mrecall.tar"),
                    self.config.model_type,
                    self.config.dataset,
                    additional_metadata={
                        "epoch": epoch,
                        "best_mrecall": self.best_Mrecall,
                        "mode": self.config.mode,
                        "enc_layer": self.config.enc_layer,
                        "dec_layer": self.config.dec_layer,
                        "checkpoint_type": "best_mrecall"
                    },
                    logger=self.logger
                )
                if checkpoint_saved:
                    self.logger.info(
                        "NEW BEST MRECALL! Saved best checkpoint after %d epochs", epoch
                    )
                else:
                    self.logger.error("Failed to save best mrecall checkpoint - disabling future saves")
                    self.config.disable_checkpoint_saving = True
            else:
                self.logger.info(
                    "NEW BEST MRECALL! Checkpoint saving disabled (epoch %d)", epoch
                )
