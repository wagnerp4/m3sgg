"""
Modularized training script for scene graph generation models.

This script provides a clean, modular interface for training various scene graph
generation models including STTran, DSG-DETR, STKET, TEMPURA, SceneLLM, OED, and VLM.
The script uses factory classes for model creation, loss function setup, and
detector instantiation to improve maintainability and extensibility.

Key Features:
- Factory-based architecture for models, losses, and detectors
- Support for multiple model types and datasets
- Comprehensive logging and checkpoint management
- Memory computation for TEMPURA models
- Hungarian matcher support for DSG-DETR

Usage:
    python scripts/training/training.py -model sttran -dataset action_genome -mode predcls
    python scripts/training/training.py -model tempura -dataset action_genome -mode predcls
    python scripts/training/training.py -model dsg-detr -dataset action_genome -mode predcls
"""

import logging
import os
import sys
import time
import warnings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from m3sgg.datasets.action_genome import cuda_collate_fn
from m3sgg.datasets.factory import get_datasets
from m3sgg.utils.AdamW import AdamW
from m3sgg.core.config.config import Config
from m3sgg.core.evaluation.metrics import BasicSceneGraphEvaluator
from m3sgg.core.training import Trainer, ModelFactory, LossFactory, MemorySetup
from m3sgg.core.detectors.factory import DetectorFactory
from m3sgg.utils.matcher import HungarianMatcher
from m3sgg.utils.util import (
    create_dataloaders,
    create_subset_samplers,
)
from m3sgg.utils.checkpoint_utils import (
    check_disk_space_and_configure_checkpointing,
    validate_checkpoint_file,
)

np.set_printoptions(precision=3)
warnings.filterwarnings("ignore", category=DeprecationWarning)

def iterative_training(trainer, conf, logger, niter):
    """Iterative training mode using train_iter with evaluation support.
    
    :param trainer: Trainer instance
    :type trainer: Trainer
    :param conf: Configuration object
    :type conf: Config
    :param logger: Logger instance
    :type logger: logging.Logger
    :param niter: Number of iterations to train
    :type niter: int
    """
    logger.info(f"Starting iterative training for {niter} iterations...")
    
    # Set evaluation frequency (default: every 50 iterations, similar to epoch-based)
    eval_frequency = conf.eval_frequency
    if niter <= eval_frequency:
        eval_frequency = niter  # Evaluate at the end if total iterations <= eval_frequency
    
    for progress in trainer.train_iter(niter):
        if progress.get("epoch_complete", False):
            logger.info("Training complete!")
            logger.info(f"Average Loss: {progress['avg_epoch_loss']:.6f}")
            logger.info(f"Loss Components: {progress['avg_epoch_loss_components']}")
        else:
            # Show progress every 10 iterations or at the end
            if progress["iteration"] % 10 == 0 or progress["iteration"] == progress["total_iterations"] - 1:
                logger.info(f"Iter {progress['iteration']+1}/{progress['total_iterations']} "
                          f"| Loss: {progress['total_loss']:.4f} "
                          f"| Progress: {progress['progress']*100:.1f}%")
            
            # Run evaluation at specified frequency
            if (progress["iteration"] + 1) % eval_frequency == 0:
                logger.info(f"Running evaluation at iteration {progress['iteration']+1}...")
                score, mrecall, predictions_data = trainer.evaluate_epoch(progress["iteration"])
                logger.info(f"Evaluation at iter {progress['iteration']+1} | R@20: {score:.4f} | MR@20: {mrecall:.4f}")
                
                # Save predictions CSV after evaluation
                if predictions_data:
                    trainer.save_predictions_csv(predictions_data)

def epoch_training(trainer, conf, logger):
    """Standard epoch-based training mode using train_loop.
    
    :param trainer: Trainer instance
    :type trainer: Trainer
    :param conf: Configuration object
    :type conf: Config
    :param logger: Logger instance
    :type logger: logging.Logger
    """
    logger.info(f"Starting epoch-based training for {int(conf.nepoch)} epochs...")
    trainer.train_loop()

def main():
    """Main training function."""
    # Settings
    # Use original config (parses command-line arguments automatically)
    conf = Config()
    seed = conf.seed
    gpu_device = torch.device(conf.device)
    new_save_path = os.path.join(
        "output",
        os.path.basename(conf.data_path),
        conf.model_type,
        conf.mode,
        f"{time.strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(new_save_path, exist_ok=True)
    conf.save_path = new_save_path

    # Logging
    log_file = os.path.join(conf.save_path, "logfile.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    logger = logging.getLogger(__name__)
    logger.info(
        "Setup | ckpt_path=%s | enc_layers=%d/dec_layers=%d | config=%s",
        conf.save_path,
        conf.enc_layer,
        conf.dec_layer,
        {k: v for k, v in conf.args.items()},
    )

    # Check disk space and configure checkpointing strategy
    checkpoint_enabled, checkpoint_strategy = check_disk_space_and_configure_checkpointing(
        conf.save_path, logger, conf
    )
    if not checkpoint_enabled:
        logger.warning("Checkpoint saving has been disabled due to insufficient disk space.")
        logger.warning("Training will continue but no checkpoints will be saved.")
    elif checkpoint_strategy == "conservative":
        logger.info("Using conservative checkpoint strategy due to low disk space.")
        logger.info("Only the best model will be saved to minimize disk usage.")

    # Dataset
    dataset_train, dataset_test = get_datasets(conf)
    train_sampler, test_sampler, train_subset_size, test_subset_size = (
        create_subset_samplers(
            len(dataset_train), len(dataset_test), fraction=conf.fraction, seed=conf.seed
        )
    )
    logger.info(f"""Using {train_subset_size}/{len(dataset_train)} 
        training samples and {test_subset_size}/{len(dataset_test)} test samples""")

    # DataLoader
    dataloader_train, dataloader_test = create_dataloaders(
        dataset_train,
        dataset_test,
        train_sampler,
        test_sampler,
        cuda_collate_fn,
        num_workers=conf.num_workers,
        pin_memory=False,
    )

    # Object detector
    detector_factory = DetectorFactory(conf, dataset_train, gpu_device, logger)
    object_detector = detector_factory.create_detector()
    
    # Maintain backward compatibility
    if conf.dataset == "EASG":
        object_detector_EASG = object_detector
        object_detector = None
    else:
        object_detector_EASG = None

    # Model Selection
    model_factory = ModelFactory(conf, dataset_train, gpu_device, logger)
    model = model_factory.create_model()

    # Checkpoint Loading
    if conf.ckpt:
        try:
            if not validate_checkpoint_file(conf.ckpt, logger):
                raise RuntimeError("Checkpoint validation failed")
            ckpt = torch.load(conf.ckpt, map_location=gpu_device)
            model.load_state_dict(ckpt["state_dict"], strict=False)
            file_size_mb = os.path.getsize(conf.ckpt) / (1024 * 1024)
            logger.info("Loaded checkpoint from: %s (%.1fMB)", conf.ckpt, file_size_mb)
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {conf.ckpt}: {str(e)}")
            logger.error("This checkpoint file appears to be corrupted or invalid.")
            logger.error("Please remove the corrupted checkpoint file and restart training without the -ckpt flag.")
            raise RuntimeError(f"Checkpoint loading failed: {str(e)}")

    # Hungarian Matcher
    if conf.use_matcher:
        matcher = HungarianMatcher(0.5, 1, 1, 0.5)
        matcher.eval()
        logger.info("Using Hungarian matcher for DSG-DETR training")
    else:
        matcher = None
        logger.info("Using default training (no matcher)")

    # Graph Evaluation
    evaluator = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=dataset_train.object_classes,
        AG_all_predicates=dataset_train.relationship_classes,
        AG_attention_predicates=dataset_train.attention_relationships,
        AG_spatial_predicates=dataset_train.spatial_relationships,
        AG_contacting_predicates=dataset_train.contacting_relationships,
        iou_threshold=0.5,
        constraint="with",
        logger=logger,
    )

    # Eval 'no' constraint
    evaluator2 = BasicSceneGraphEvaluator(
        mode=conf.mode,
        AG_object_classes=dataset_train.object_classes,
        AG_all_predicates=dataset_train.relationship_classes,
        AG_attention_predicates=dataset_train.attention_relationships,
        AG_spatial_predicates=dataset_train.spatial_relationships,
        AG_contacting_predicates=dataset_train.contacting_relationships,
        iou_threshold=0.5,
        constraint="no",
        logger=logger,
    )

    # Loss Function
    if conf.bce_loss:
        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCELoss()
    else:
        ce_loss = nn.CrossEntropyLoss()
        mlm_loss = nn.MultiLabelMarginLoss()

    # Optimizer
    if conf.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=conf.lr)
    elif conf.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01
        )

    # Loss Function Setup
    loss_factory = LossFactory(conf, model, gpu_device, logger)
    losses = loss_factory.create_losses()
    
    # Extract individual loss functions for backward compatibility
    ce_loss = losses["ce_loss"]
    bce_loss = losses.get("bce_loss")
    mlm_loss = losses.get("mlm_loss")
    ce_loss_obj = losses.get("ce_loss_obj")
    ce_loss_rel = losses.get("ce_loss_rel")
    con_loss = losses.get("con_loss")

    # Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        "max",
        patience=1,
        factor=0.5,
        threshold=1e-4,
        threshold_mode="abs",
        min_lr=1e-7,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        config=conf,
        dataloader_train=dataloader_train,
        dataloader_test=dataloader_test,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        object_detector=object_detector,
        object_detector_EASG=object_detector_EASG,
        matcher=matcher,
        evaluator=evaluator,
        evaluator2=evaluator2,
        dataset_train=dataset_train,
        dataset_test=dataset_test,
    )
    
    # Choose training mode
    if conf.niter is not None:
        iterative_training(trainer, conf, logger, conf.niter)
    else:
        epoch_training(trainer, conf, logger)
    
    logger.info("Training completed!")
    logger.info(
        "Best model achieved at epoch %d with R@20 score: %.4f",
        trainer.best_epoch,
        trainer.best_score,
    )

    # Memory Setup (for TEMPURA models)
    memory_setup = MemorySetup(conf, model, gpu_device, logger)
    memory_configured = memory_setup.setup_memory()
    
    if memory_configured:
        memory_info = memory_setup.get_memory_info()
        logger.info(f"Memory setup completed: {memory_info}")


if __name__ == "__main__":
    main()
