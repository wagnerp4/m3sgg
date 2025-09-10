"""
Model factory for creating different model types based on configuration.

This module contains the model instantiation logic that was extracted from
the monolithic training.py script to improve modularity and maintainability.
"""

import json
import logging
from typing import Optional

import torch

from m3sgg.core.detectors.easg.sttran_EASG import STTran as STTran_EASG
from m3sgg.core.models.sttran import STTran
from m3sgg.core.models.stket import STKET
from m3sgg.core.models.tempura.tempura import TEMPURA


class ModelFactory:
    """Factory class for creating model instances based on configuration.

    This factory provides a unified interface for creating different scene graph
    generation models including STTran, DSG-DETR, STKET, TEMPURA, SceneLLM, OED,
    and VLM models. It handles model instantiation logic that was extracted from
    the monolithic training script to improve modularity and maintainability.

    :param config: Configuration object containing model parameters
    :type config: Config
    :param dataset_train: Training dataset for extracting class information
    :type dataset_train: Dataset
    :param device: Device to place the model on
    :type device: torch.device
    :param logger: Optional logger instance
    :type logger: Optional[logging.Logger]
    """

    def __init__(
        self, config, dataset_train, device, logger: Optional[logging.Logger] = None
    ):
        """Initialize the model factory.

        :param config: Configuration object containing model parameters
        :type config: Config
        :param dataset_train: Training dataset for extracting class information
        :type dataset_train: Dataset
        :param device: Device to place the model on
        :type device: torch.device
        :param logger: Optional logger instance
        :type logger: Optional[logging.Logger]
        """
        self.config = config
        self.dataset_train = dataset_train
        self.device = device
        self.logger = logger or logging.getLogger(__name__)

    def create_model(self) -> torch.nn.Module:
        """Create a model instance based on the configuration.

        :return: Instantiated model
        :rtype: torch.nn.Module
        :raises ValueError: If dataset or model type is not supported
        """
        if self.config.dataset == "EASG":
            return self._create_easg_model()
        elif self.config.dataset == "action_genome":
            return self._create_action_genome_model()
        else:
            raise ValueError(f"Dataset '{self.config.dataset}' not supported")

    def _create_easg_model(self) -> STTran_EASG:
        """Create STTran model for EASG dataset.

        :return: STTran_EASG model instance
        :rtype: STTran_EASG
        """
        model = STTran_EASG(
            mode=self.config.mode,
            obj_classes=self.dataset_train.obj_classes,
            verb_classes=self.dataset_train.verb_classes,
            edge_class_num=len(self.dataset_train.edge_classes),
            enc_layer_num=self.config.enc_layer,
            dec_layer_num=self.config.dec_layer,
        ).to(device=self.device)

        self.logger.info("Created STTran_EASG model for EASG dataset")
        return model

    def _create_action_genome_model(self) -> torch.nn.Module:
        """Create model for Action Genome dataset based on model type.

        :return: Model instance
        :rtype: torch.nn.Module
        :raises ValueError: If model type is not supported
        """
        if self.config.model_type == "sttran":
            return self._create_sttran_model()
        elif self.config.model_type == "dsg-detr":
            return self._create_dsg_detr_model()
        elif self.config.model_type == "stket":
            return self._create_stket_model()
        elif self.config.model_type == "tempura":
            return self._create_tempura_model()
        elif self.config.model_type == "scenellm":
            return self._create_scenellm_model()
        elif self.config.model_type == "oed":
            return self._create_oed_model()
        elif self.config.model_type == "vlm":
            return self._create_vlm_model()
        else:
            raise ValueError(
                f"Model type '{self.config.model_type}' not supported for Action Genome dataset"
            )

    def _create_sttran_model(self) -> STTran:
        """Create STTran model.

        :return: STTran model instance
        :rtype: STTran
        """
        model = STTran(
            mode=self.config.mode,
            attention_class_num=len(self.dataset_train.attention_relationships),
            spatial_class_num=len(self.dataset_train.spatial_relationships),
            contact_class_num=len(self.dataset_train.contacting_relationships),
            obj_classes=self.dataset_train.object_classes,
            enc_layer_num=self.config.enc_layer,
            dec_layer_num=self.config.dec_layer,
        ).to(device=self.device)

        self.logger.info("Created STTran model")
        return model

    def _create_dsg_detr_model(self) -> STTran:
        """Create DSG-DETR model (STTran with matcher).

        :return: STTran model instance configured for DSG-DETR
        :rtype: STTran
        """
        self.config.use_matcher = True
        model = STTran(
            mode=self.config.mode,
            attention_class_num=len(self.dataset_train.attention_relationships),
            spatial_class_num=len(self.dataset_train.spatial_relationships),
            contact_class_num=len(self.dataset_train.contacting_relationships),
            obj_classes=self.dataset_train.object_classes,
            enc_layer_num=self.config.enc_layer,
            dec_layer_num=self.config.dec_layer,
        ).to(device=self.device)

        self.logger.info("Created DSG-DETR model (STTran with matcher)")
        return model

    def _create_stket_model(self) -> STKET:
        """Create STKET model.

        :return: STKET model instance
        :rtype: STKET
        """
        train_prior = (
            json.load(open("data/TrainPrior.json", "r"))
            if self.config.model_type == "stket"
            else None
        )

        model = STKET(
            mode=self.config.mode,
            attention_class_num=len(self.dataset_train.attention_relationships),
            spatial_class_num=len(self.dataset_train.spatial_relationships),
            contact_class_num=len(self.dataset_train.contacting_relationships),
            obj_classes=self.dataset_train.object_classes,
            N_layer_num=self.config.N_layer,
            enc_layer_num=self.config.enc_layer_num,
            dec_layer_num=self.config.dec_layer_num,
            pred_contact_threshold=self.config.pred_contact_threshold,
            window_size=self.config.window_size,
            trainPrior=train_prior,
            use_spatial_prior=self.config.use_spatial_prior,
            use_temporal_prior=self.config.use_temporal_prior,
        ).to(device=self.device)

        self.logger.info("Created STKET model")
        return model

    def _create_tempura_model(self) -> TEMPURA:
        """Create TEMPURA model.

        :return: TEMPURA model instance
        :rtype: TEMPURA
        """
        model = TEMPURA(
            mode=self.config.mode,
            attention_class_num=len(self.dataset_train.attention_relationships),
            spatial_class_num=len(self.dataset_train.spatial_relationships),
            contact_class_num=len(self.dataset_train.contacting_relationships),
            obj_classes=self.dataset_train.object_classes,
            enc_layer_num=self.config.enc_layer,
            dec_layer_num=self.config.dec_layer,
            obj_mem_compute=self.config.obj_mem_compute,
            rel_mem_compute=self.config.rel_mem_compute,
            take_obj_mem_feat=self.config.take_obj_mem_feat,
            mem_fusion=self.config.mem_fusion,
            selection=self.config.mem_feat_selection,
            selection_lambda=self.config.mem_feat_lambda,
            obj_head=self.config.obj_head,
            rel_head=self.config.rel_head,
            K=self.config.K,
        ).to(device=self.device)

        self.logger.info("Created TEMPURA model")
        return model

    def _create_scenellm_model(self) -> torch.nn.Module:
        """Create SceneLLM model.

        :return: SceneLLM model instance
        :rtype: torch.nn.Module
        :raises RuntimeError: If SceneLLM import fails
        """
        try:
            from lib.scenellm.scenellm import SceneLLM

            model = SceneLLM(self.config, self.dataset_train).to(device=self.device)
            model.set_training_stage(self.config.scenellm_training_stage)
            self.logger.info(
                f"Initialized SceneLLM with training stage: {self.config.scenellm_training_stage}"
            )
            return model
        except ImportError as e:
            self.logger.error(f"Failed to import SceneLLM: {e}")
            self.logger.error(
                "SceneLLM requires transformers and peft packages. Please install them or use a different model."
            )
            raise RuntimeError(f"SceneLLM import failed: {e}")

    def _create_oed_model(self) -> torch.nn.Module:
        """Create OED model (Multi or Single frame).

        :return: OED model instance
        :rtype: torch.nn.Module
        """
        from lib.oed import OEDMulti, OEDSingle

        if self.config.oed_variant == "multi":
            model = OEDMulti(self.config, self.dataset_train).to(device=self.device)
            self.logger.info("Initialized OED Multi-frame model")
        else:
            model = OEDSingle(self.config, self.dataset_train).to(device=self.device)
            self.logger.info("Initialized OED Single-frame model")

        return model

    def _create_vlm_model(self) -> torch.nn.Module:
        """Create VLM Scene Graph Generator model.

        :return: VLM model instance
        :rtype: torch.nn.Module
        """
        from m3sgg.core.models.vlm import VLMSceneGraphGenerator

        model = VLMSceneGraphGenerator(
            mode=self.config.mode,
            attention_class_num=len(self.dataset_train.attention_relationships),
            spatial_class_num=len(self.dataset_train.spatial_relationships),
            contact_class_num=len(self.dataset_train.contacting_relationships),
            obj_classes=self.dataset_train.object_classes,
            model_name=self.config.vlm_model_name,
            device=self.device,
            use_chain_of_thought=self.config.vlm_use_chain_of_thought,
            use_tree_of_thought=self.config.vlm_use_tree_of_thought,
            confidence_threshold=self.config.vlm_confidence_threshold,
        ).to(device=self.device)

        self.logger.info(f"Initialized VLM model: {self.config.vlm_model_name}")
        return model
