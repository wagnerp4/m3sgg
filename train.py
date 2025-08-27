import copy
import json
import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim

# from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from dataloader.action_genome import AG, cuda_collate_fn
from dataloader.easg import EASG
from lib.AdamW import AdamW
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.infoNCE import EucNormLoss, SupConLoss
from lib.matcher import HungarianMatcher
from lib.memory import memory_computation
from lib.object_detector import detector
from lib.object_detector_EASG import detector as detector_EASG
from lib.scenellm.scenellm import SceneLLM
from lib.sttran import STKET, STTran
from lib.sttran_EASG import STTran as STTran_EASG
from lib.tempura.tempura import TEMPURA
from lib.track import get_sequence
from lib.uncertainty import uncertainty_computation, uncertainty_values
from utils.util import (
    create_subset_samplers,
    get_pred_triplets,
    intersect_2d,
)

np.set_printoptions(precision=3)
warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == "__main__":
    conf = Config()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    data_path_suffix = os.path.basename(conf.data_path)
    new_save_path = os.path.join(
        "output", data_path_suffix, conf.model_type, conf.mode, f"{timestamp}"
    )  # conf.dataset,
    os.makedirs(new_save_path, exist_ok=True)
    conf.save_path = new_save_path

    # Setup logging
    log_file = os.path.join(conf.save_path, "logfile.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger(__name__)
    logger.info("The CKPT saved here: %s", conf.save_path)
    logger.info(
        "spatial encoder layer num: %d / temporal decoder layer num: %d",
        conf.enc_layer,
        conf.dec_layer,
    )
    logger.info("Config arguments:")
    for i in conf.args:
        logger.info("%s: %s", i, conf.args[i])

    # Dataset
    if conf.dataset == "EASG":
        dataset_train = EASG(
            split="train",
            datasize=conf.datasize,
            data_path=conf.data_path,
        )
        dataset_test = EASG(
            split="val",
            datasize=conf.datasize,
            data_path=conf.data_path,
        )
    elif conf.dataset == "action_genome":
        dataset_train = AG(
            mode="train",
            datasize=conf.datasize,
            data_path=conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == "predcls" else True,
        )
        dataset_test = AG(
            mode="test",
            datasize=conf.datasize,
            data_path=conf.data_path,
            filter_nonperson_box_frame=True,
            filter_small_box=False if conf.mode == "predcls" else True,
        )

    # Subset Selection
    train_sampler, test_sampler, train_subset_size, test_subset_size = (
        create_subset_samplers(
            len(dataset_train), len(dataset_test), fraction=1, seed=42
        )
    )

    # DataLoader
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        # shuffle=True,
        sampler=train_sampler,
        num_workers=0,
        collate_fn=cuda_collate_fn,
        pin_memory=False,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        # shuffle=False,
        sampler=test_sampler,
        num_workers=0,
        collate_fn=cuda_collate_fn,
        pin_memory=False,
    )

    logger.info(
        f"Using {train_subset_size}/{len(dataset_train)} training samples and {test_subset_size}/{len(dataset_test)} test samples"
    )

    gpu_device = torch.device("cuda:0")

    # Object detector
    if conf.dataset == "EASG":
        object_detector_EASG = detector_EASG(
            train=True,
            object_classes=dataset_train.obj_classes,
            use_SUPPLY=True,
            mode=conf.mode,
        ).to(device=gpu_device)
        object_detector_EASG.eval()
    elif conf.dataset == "action_genome":
        object_detector = detector(  # freeze the detection backbone
            train=True,
            object_classes=dataset_train.object_classes,
            use_SUPPLY=True,
            mode=conf.mode,
        ).to(device=gpu_device)
        object_detector.eval()

    # Model Selection - Dataset-dependent
    if conf.dataset == "EASG":
        # TODO: Implement EASG-specific model
        model = STTran_EASG(
            mode=conf.mode,
            obj_classes=dataset_train.obj_classes,
            verb_classes=dataset_train.verb_classes,
            edge_class_num=len(dataset_train.edge_classes),
            enc_layer_num=conf.enc_layer,
            dec_layer_num=conf.dec_layer,
        ).to(device=gpu_device)
    elif conf.dataset == "action_genome":
        if conf.model_type == "sttran":
            model = STTran(
                mode=conf.mode,
                attention_class_num=len(dataset_train.attention_relationships),
                spatial_class_num=len(dataset_train.spatial_relationships),
                contact_class_num=len(dataset_train.contacting_relationships),
                obj_classes=dataset_train.object_classes,
                enc_layer_num=conf.enc_layer,
                dec_layer_num=conf.dec_layer,
            ).to(device=gpu_device)
        elif conf.model_type == "dsg-detr":
            conf.use_matcher = True
            model = STTran(
                mode=conf.mode,
                attention_class_num=len(dataset_train.attention_relationships),
                spatial_class_num=len(dataset_train.spatial_relationships),
                contact_class_num=len(dataset_train.contacting_relationships),
                obj_classes=dataset_train.object_classes,
                enc_layer_num=conf.enc_layer,
                dec_layer_num=conf.dec_layer,
            ).to(device=gpu_device)
        elif conf.model_type == "stket":
            trainPrior = (
                json.load(open("data/TrainPrior.json", "r"))
                if conf.model_type == "stket"
                else None
            )
            model = STKET(
                mode=conf.mode,
                attention_class_num=len(dataset_train.attention_relationships),
                spatial_class_num=len(dataset_train.spatial_relationships),
                contact_class_num=len(dataset_train.contacting_relationships),
                obj_classes=dataset_train.object_classes,
                N_layer_num=conf.N_layer,
                enc_layer_num=conf.enc_layer_num,
                dec_layer_num=conf.dec_layer_num,
                pred_contact_threshold=conf.pred_contact_threshold,
                window_size=conf.window_size,
                trainPrior=trainPrior,
                use_spatial_prior=conf.use_spatial_prior,
                use_temporal_prior=conf.use_temporal_prior,
            ).to(device=gpu_device)
        elif conf.model_type == "tempura":
            model = TEMPURA(
                mode=conf.mode,
                attention_class_num=len(dataset_train.attention_relationships),
                spatial_class_num=len(dataset_train.spatial_relationships),
                contact_class_num=len(dataset_train.contacting_relationships),
                obj_classes=dataset_train.object_classes,
                enc_layer_num=conf.enc_layer,
                dec_layer_num=conf.dec_layer,
                obj_mem_compute=conf.obj_mem_compute,
                rel_mem_compute=conf.rel_mem_compute,
                take_obj_mem_feat=conf.take_obj_mem_feat,
                mem_fusion=conf.mem_fusion,
                selection=conf.mem_feat_selection,
                selection_lambda=conf.mem_feat_lambda,
                obj_head=conf.obj_head,
                rel_head=conf.rel_head,
                K=conf.K,
            ).to(device=gpu_device)
        elif conf.model_type == "scenellm":
            model = SceneLLM(conf, dataset_train).to(device=gpu_device)
            print(
                f"DEBUG: About to set training stage to: {conf.scenellm_training_stage}"
            )
            model.set_training_stage(conf.scenellm_training_stage)
            print(f"DEBUG: Model training stage is now: {model.training_stage}")
            logger.info(
                f"SceneLLM initialized with training stage: {conf.scenellm_training_stage}"
            )
        else:
            raise ValueError(
                f"Model type '{conf.model_type}' not supported for Action Genome dataset"
            )
    else:
        raise ValueError(f"Dataset '{conf.dataset}' not supported")

    if conf.ckpt:
        ckpt = torch.load(conf.ckpt, map_location=gpu_device)
        model.load_state_dict(ckpt["state_dict"], strict=False)
        logger.info("Loaded checkpoint from: %s", conf.ckpt)

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
    # TODO: no constraint Graph Evaluation
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

    if conf.bce_loss:
        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCELoss()
    else:
        ce_loss = nn.CrossEntropyLoss()
        mlm_loss = nn.MultiLabelMarginLoss()

    if conf.optimizer == "adamw":
        optimizer = AdamW(model.parameters(), lr=conf.lr)
    elif conf.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01
        )

    if conf.model_type == "tempura":
        weights = torch.ones(len(model.obj_classes))
        weights[0] = conf.eos_coef
        if conf.obj_head != "gmm":
            ce_loss_obj = nn.CrossEntropyLoss(
                weight=weights.to(device=gpu_device), reduction="none"
            )
        else:
            ce_loss_obj = nn.NLLLoss(
                weight=weights.to(device=gpu_device), reduction="none"
            )
        if conf.rel_head != "gmm":
            ce_loss_rel = nn.CrossEntropyLoss(reduction="none")
        else:
            ce_loss_rel = nn.NLLLoss(reduction="none")
        if conf.mlm:
            mlm_loss = nn.MultiLabelMarginLoss(reduction="none")
        else:
            bce_loss = nn.BCELoss(reduction="none")
        if conf.obj_con_loss == "euc_con":
            # con_loss = metric_loss.ContrastiveLoss(pos_margin=0, neg_margin=1)
            con_loss = EucNormLoss()
            con_loss.train()
        elif conf.obj_con_loss == "info_nce":
            con_loss = SupConLoss(temperature=0.1)
            con_loss.train()

    # STTran
    # scheduler = ReduceLROnPlateau(
    #         optimizer, "max", patience=1, factor=0.5, verbose=True,
    #         threshold=1e-4, threshold_mode="abs",
    #         min_lr=1e-7)

    # DSG-DETR
    scheduler = ReduceLROnPlateau(
        optimizer,
        "max",
        patience=1,
        factor=0.5,
        verbose=True,
        threshold=1e-4,
        threshold_mode="abs",
        min_lr=1e-7,
    )

    # STKET
    # scheduler = ReduceLROnPlateau(
    #     optimizer, "max", patience=0, factor=0.5,
    #     verbose=True, threshold=1e-3, threshold_mode="abs",
    #     min_lr=1e-7)

    # unbiasedSGG
    # scheduler = ReduceLROnPlateau(
    #     optimizer, "max", patience=1, factor=0.5, verbose=True,
    #     threshold=1e-4, threshold_mode="abs", min_lr=1e-7)

    logger.info("Using %s optimizer with learning rate %e", conf.optimizer, conf.lr)
    logger.info("Starting training...")

    tr = []
    if conf.use_matcher:
        matcher = HungarianMatcher(0.5, 1, 1, 0.5)
        matcher.eval()
        logger.info("Using Hungarian matcher for DSG-DETR training")
    else:
        matcher = None
        logger.info("Using default STTran training (no matcher)")

    logger.info("Training dataset size: %d batches", len(dataloader_train))
    logger.info("Test dataset size: %d batches", len(dataloader_test))

    print(type(conf.nepoch))
    conf.nepoch = int(conf.nepoch)
    best_score = 0.0
    best_Mrecall = 0.0
    best_epoch = -1

    # Training loop
    for epoch in range(conf.nepoch):
        logger.info("=" * 40)
        logger.info("Starting Epoch %d", epoch)
        logger.info("=" * 40)

        if conf.model_type == "tempura":
            unc_vals = uncertainty_values(
                obj_classes=len(model.obj_classes),
                attention_class_num=model.attention_class_num,
                spatial_class_num=model.spatial_class_num,
                contact_class_num=model.contact_class_num,
            )

        model.train()
        if conf.dataset == "action_genome":
            object_detector.is_train = True
            object_detector.train_x = True
        elif conf.dataset == "EASG":
            object_detector_EASG.is_train = True
        else:
            raise ValueError(f"Dataset '{conf.dataset}' not supported")

        start = time.time()
        train_iter = iter(dataloader_train)
        test_iter = iter(dataloader_test)

        epoch_train_losses = []
        epoch_train_loss_components = []

        train_pbar = tqdm(
            range(len(dataloader_train)), desc=f"Epoch {epoch}/{conf.nepoch} [Train]"
        )

        for b in train_pbar:
            data = next(train_iter)
            im_data = copy.deepcopy(data[0].cuda(0))
            im_info = copy.deepcopy(data[1].cuda(0))

            # Dataset-specific data processing
            if conf.dataset == "EASG":
                gt_grounding = dataset_train.gt_groundings[data[2]]
                with torch.no_grad():
                    entry = object_detector_EASG(
                        im_data, im_info, gt_grounding, im_all=None
                    )
                entry["features_verb"] = copy.deepcopy(
                    dataset_train.verb_feats[data[2]].cuda(0)
                )
            else:
                gt_boxes = copy.deepcopy(data[2].cuda(0))
                num_boxes = copy.deepcopy(data[3].cuda(0))
                gt_annotation = dataset_train.gt_annotations[data[4]]
                with torch.no_grad():
                    entry = object_detector(
                        im_data,
                        im_info,
                        gt_boxes,
                        num_boxes,
                        gt_annotation,
                        im_all=None,
                    )

            if conf.use_matcher and conf.dataset != "EASG":
                get_sequence(
                    entry,
                    gt_annotation,
                    matcher,
                    (im_info[0][:2] / im_info[0, 2]).cpu().data,
                    conf.mode,
                )

            pred = model(entry)

            if conf.model_type == "scenellm":
                # Update codebook with OT scheme periodically (every 1000 batches)
                if b > 0 and b % 1000 == 0 and conf.scenellm_training_stage != "vqvae":
                    model.update_codebook_with_ot()

            if (
                conf.obj_unc
                or conf.rel_unc
                or conf.obj_mem_compute
                or conf.rel_mem_compute
            ):
                uncertainty_computation(
                    data,
                    dataset_train,
                    object_detector,
                    model,
                    unc_vals,
                    gpu_device,
                    conf.save_path,
                    obj_unc=conf.obj_unc,
                    obj_mem=conf.obj_mem_compute,
                    background_mem=False,
                    rel_unc=conf.rel_unc,
                )

            # Dataset-specific loss calculations
            if conf.dataset == "EASG":
                edge_distribution = pred["edge_distribution"]
                losses = {}
                if conf.mode != "edgecls":
                    losses["obj_loss"] = ce_loss(pred["distribution"], pred["labels"])
                if conf.mode == "easgcls":
                    losses["verb_loss"] = ce_loss(
                        pred["distribution_verb"], pred["labels_verb"]
                    )
                edge_label = -torch.ones(
                    [len(pred["edge"]), len(dataset_train.edge_classes)],
                    dtype=torch.long,
                ).to(device=edge_distribution.device)
                for i in range(len(pred["edge"])):
                    edge_label[i, : len(pred["edge"][i])] = torch.tensor(
                        pred["edge"][i]
                    )
                losses["edge_loss"] = mlm_loss(edge_distribution, edge_label)
            elif conf.dataset == "action_genome":
                if conf.model_type in ["sttran", "dsg-detr"]:
                    attention_distribution = pred["attention_distribution"]
                    spatial_distribution = pred["spatial_distribution"]
                    contact_distribution = pred["contact_distribution"]
                    attention_label = (
                        torch.tensor(pred["attention_gt"], dtype=torch.long)
                        .to(device=attention_distribution.device)
                        .squeeze()
                    )
                    if not conf.bce_loss:
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
                    if conf.mode == "sgcls" or conf.mode == "sgdet":
                        losses["object_loss"] = ce_loss(
                            pred["distribution"], pred["labels"]
                        )

                    losses["attention_relation_loss"] = ce_loss(
                        attention_distribution, attention_label
                    )
                    if not conf.bce_loss:
                        losses["spatial_relation_loss"] = mlm_loss(
                            spatial_distribution, spatial_label
                        )
                        losses["contact_relation_loss"] = mlm_loss(
                            contact_distribution, contact_label
                        )
                    else:
                        losses["spatial_relation_loss"] = bce_loss(
                            spatial_distribution, spatial_label
                        )
                        losses["contact_relation_loss"] = bce_loss(
                            contact_distribution, contact_label
                        )
                elif conf.model_type == "stket":
                    attention_label = (
                        torch.tensor(pred["attention_gt"], dtype=torch.long)
                        .to(device=im_data.device)
                        .squeeze()
                    )
                    if not conf.bce_loss:
                        # multi-label margin loss or adaptive loss
                        spatial_label = -torch.ones(
                            [len(pred["spatial_gt"]), 6], dtype=torch.long
                        ).to(device=im_data.device)
                        contact_label = -torch.ones(
                            [len(pred["contact_gt"]), 17], dtype=torch.long
                        ).to(device=im_data.device)
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
                        ).to(device=im_data.device)
                        contact_label = torch.zeros(
                            [len(pred["contact_gt"]), 17], dtype=torch.float32
                        ).to(device=im_data.device)
                        for i in range(len(pred["spatial_gt"])):
                            spatial_label[i, pred["spatial_gt"][i]] = 1
                            contact_label[i, pred["contact_gt"][i]] = 1

                    losses = {}
                    if conf.mode == "sgcls" or conf.mode == "sgdet":
                        losses["object_loss"] = ce_loss(
                            pred["distribution"], pred["labels"]
                        )

                    # Spatial encoder losses
                    if conf.enc_layer_num > 0:
                        losses["spatial_attention_relation_loss"] = ce_loss(
                            pred["spatial_attention_distribution"], attention_label
                        )
                        if not conf.bce_loss:
                            losses["spatial_spatial_relation_loss"] = mlm_loss(
                                pred["spatial_spatial_distribution"], spatial_label
                            )
                            losses["spatial_contact_relation_loss"] = mlm_loss(
                                pred["spatial_contact_distribution"], contact_label
                            )
                        else:
                            losses["spatial_spatial_relation_loss"] = bce_loss(
                                pred["spatial_spatial_distribution"], spatial_label
                            )
                            losses["spatial_contact_relation_loss"] = bce_loss(
                                pred["spatial_contact_distribution"], contact_label
                            )

                    # Temporal decoder losses
                    if conf.dec_layer_num > 0:
                        losses["temporal_attention_relation_loss"] = ce_loss(
                            pred["temporal_attention_distribution"], attention_label
                        )
                        if not conf.bce_loss:
                            losses["temporal_spatial_relation_loss"] = mlm_loss(
                                pred["temporal_spatial_distribution"], spatial_label
                            )
                            losses["temporal_contact_relation_loss"] = mlm_loss(
                                pred["temporal_contact_distribution"], contact_label
                            )
                        else:
                            losses["temporal_spatial_relation_loss"] = bce_loss(
                                pred["temporal_spatial_distribution"], spatial_label
                            )
                            losses["temporal_contact_relation_loss"] = bce_loss(
                                pred["temporal_contact_distribution"], contact_label
                            )

                    # Ensemble losses
                    if (conf.enc_layer_num > 0) and (conf.dec_layer_num > 0):
                        losses["ensemble_attention_relation_loss"] = ce_loss(
                            pred["ensemble_attention_distribution"], attention_label
                        )
                        if not conf.bce_loss:
                            losses["ensemble_spatial_relation_loss"] = mlm_loss(
                                pred["ensemble_spatial_distribution"], spatial_label
                            )
                            losses["ensemble_contact_relation_loss"] = mlm_loss(
                                pred["ensemble_contact_distribution"], contact_label
                            )
                        else:
                            losses["ensemble_spatial_relation_loss"] = bce_loss(
                                pred["ensemble_spatial_distribution"], spatial_label
                            )
                            losses["ensemble_contact_relation_loss"] = bce_loss(
                                pred["ensemble_contact_distribution"], contact_label
                            )

                    # Prior losses if enabled
                    if conf.use_spatial_prior and conf.spatial_prior_loss:
                        losses["spatial_prior_attention_relation_loss"] = ce_loss(
                            pred["spatial_prior_attention_distribution"],
                            attention_label,
                        )
                        if not conf.bce_loss:
                            losses["spatial_prior_spatial_relation_loss"] = mlm_loss(
                                pred["spatial_prior_spatial_distribution"],
                                spatial_label,
                            )
                            losses["spatial_prior_contact_relation_loss"] = mlm_loss(
                                pred["spatial_prior_contact_distribution"],
                                contact_label,
                            )
                        else:
                            losses["spatial_prior_spatial_relation_loss"] = bce_loss(
                                pred["spatial_prior_spatial_distribution"],
                                spatial_label,
                            )
                            losses["spatial_prior_contact_relation_loss"] = bce_loss(
                                pred["spatial_prior_contact_distribution"],
                                contact_label,
                            )

                    if conf.use_temporal_prior and conf.temporal_prior_loss:
                        losses["temporal_prior_attention_relation_loss"] = ce_loss(
                            pred["temporal_prior_attention_distribution"],
                            attention_label,
                        )
                        if not conf.bce_loss:
                            losses["temporal_prior_spatial_relation_loss"] = mlm_loss(
                                pred["temporal_prior_spatial_distribution"],
                                spatial_label,
                            )
                            losses["temporal_prior_contact_relation_loss"] = mlm_loss(
                                pred["temporal_prior_contact_distribution"],
                                contact_label,
                            )
                        else:
                            losses["temporal_prior_spatial_relation_loss"] = bce_loss(
                                pred["temporal_prior_spatial_distribution"],
                                spatial_label,
                            )
                            losses["temporal_prior_contact_relation_loss"] = bce_loss(
                                pred["temporal_prior_contact_distribution"],
                                contact_label,
                            )
                elif conf.model_type == "tempura":
                    attention_distribution = pred["attention_distribution"]
                    spatial_distribution = pred["spatial_distribution"]
                    contact_distribution = pred["contacting_distribution"]

                    if conf.rel_head == "gmm":
                        attention_distribution = torch.log(
                            attention_distribution + 1e-12
                        )

                    if conf.obj_head == "gmm" and conf.mode != "predcls":
                        pred["distribution"] = torch.log(pred["distribution"] + 1e-12)

                    attention_label = (
                        torch.tensor(pred["attention_gt"], dtype=torch.long)
                        .to(device=attention_distribution.device)
                        .squeeze()
                    )
                    if conf.mlm:
                        # multi-label margin loss or adaptive loss
                        spatial_label = -torch.ones(
                            [len(pred["spatial_gt"]), 6], dtype=torch.long
                        ).to(device=attention_distribution.device)
                        contact_label = -torch.ones(
                            [len(pred["contacting_gt"]), 17], dtype=torch.long
                        ).to(device=attention_distribution.device)
                        for i in range(len(pred["spatial_gt"])):
                            spatial_label[i, : len(pred["spatial_gt"][i])] = (
                                torch.tensor(pred["spatial_gt"][i])
                            )
                            contact_label[i, : len(pred["contacting_gt"][i])] = (
                                torch.tensor(pred["contacting_gt"][i])
                            )
                    else:
                        # bce loss
                        spatial_label = torch.zeros(
                            [len(pred["spatial_gt"]), 6], dtype=torch.float32
                        ).to(device=attention_distribution.device)
                        contact_label = torch.zeros(
                            [len(pred["contacting_gt"]), 17], dtype=torch.float32
                        ).to(device=attention_distribution.device)
                        for i in range(len(pred["spatial_gt"])):
                            spatial_label[i, pred["spatial_gt"][i]] = 1
                            contact_label[i, pred["contacting_gt"][i]] = 1

                    losses = {}
                    if conf.mode == "sgcls" or conf.mode == "sgdet":
                        losses["object_loss"] = ce_loss_obj(
                            pred["distribution"], pred["labels"]
                        )
                        loss_weighting = conf.obj_loss_weighting
                        if loss_weighting is not None:
                            num = torch.exp(
                                unc_vals.obj_batch_unc[loss_weighting].sum(-1)
                            )
                            den = num.sum()
                            weights = 1 + (num / den).to(device=gpu_device)
                            losses["object_loss"] = weights * losses["object_loss"]
                        losses["object_loss"] = losses["object_loss"].mean()
                        if conf.obj_con_loss:
                            losses["object_contrastive_loss"] = (
                                conf.lambda_con
                                * con_loss(pred["object_mem_features"], pred["labels"])
                            )

                    losses["attention_relation_loss"] = ce_loss_rel(
                        attention_distribution, attention_label
                    )
                    if conf.mlm:
                        losses["spatial_relation_loss"] = mlm_loss(
                            spatial_distribution, spatial_label
                        )
                        losses["contacting_relation_loss"] = mlm_loss(
                            contact_distribution, contact_label
                        )
                    else:
                        losses["spatial_relation_loss"] = bce_loss(
                            spatial_distribution, spatial_label
                        )
                        losses["contacting_relation_loss"] = bce_loss(
                            contact_distribution, contact_label
                        )

                    loss_weighting = conf.rel_loss_weighting

                    for rel in ["attention", "spatial", "contacting"]:
                        if loss_weighting is not None:
                            num = torch.exp(
                                unc_vals.rel_batch_unc[rel][loss_weighting].sum(-1)
                            )
                            den = num.sum() + 1e-12
                            weights = 1 + (num / den).to(device=gpu_device)

                            if rel != "attention":
                                weights = weights.unsqueeze(-1).repeat(
                                    1, losses[rel + "_relation_loss"].shape[-1]
                                )

                            losses[rel + "_relation_loss"] = (
                                weights * losses[rel + "_relation_loss"]
                            )
                        losses[rel + "_relation_loss"] = losses[
                            rel + "_relation_loss"
                        ].mean()
                elif conf.model_type == "scenellm":
                    # SceneLLM loss computation
                    attention_distribution = pred["attention_distribution"]
                    spatial_distribution = pred["spatial_distribution"]
                    contact_distribution = pred["contact_distribution"]
                    attention_label = (
                        torch.tensor(pred["attention_gt"], dtype=torch.long)
                        .to(device=attention_distribution.device)
                        .squeeze()
                    )

                    # Handle spatial and contact labels
                    if not conf.bce_loss:
                        # multi-label margin loss
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

                    # VQ-VAE losses (only during VQ-VAE pretraining)
                    if conf.scenellm_training_stage == "vqvae":
                        losses["vq_loss"] = pred["vq_loss"]
                        losses["recon_loss"] = pred["recon_loss"]
                        losses["embedding_loss"] = pred["embedding_loss"]
                        losses["commitment_loss"] = pred["commitment_loss"]
                    else:
                        # SGG losses for stage 1 and stage 2
                        if conf.mode == "sgcls" or conf.mode == "sgdet":
                            losses["object_loss"] = conf.alpha_obj * ce_loss(
                                pred["distribution"], pred["labels"]
                            )

                        losses["attention_relation_loss"] = conf.alpha_rel * ce_loss(
                            attention_distribution, attention_label
                        )

                        if not conf.bce_loss:
                            losses["spatial_relation_loss"] = conf.alpha_rel * mlm_loss(
                                spatial_distribution, spatial_label
                            )
                            losses["contact_relation_loss"] = conf.alpha_rel * mlm_loss(
                                contact_distribution, contact_label
                            )
                        else:
                            losses["spatial_relation_loss"] = conf.alpha_rel * bce_loss(
                                spatial_distribution, spatial_label
                            )
                            losses["contact_relation_loss"] = conf.alpha_rel * bce_loss(
                                contact_distribution, contact_label
                            )

                        # Add VQ-VAE regularization loss (smaller weight)
                        if "vq_loss" in pred:
                            losses["vq_regularization"] = 0.1 * pred["vq_loss"]
                else:
                    raise ValueError(f"Model type '{conf.model_type}' not supported")
            else:
                raise ValueError(f"Dataset '{conf.dataset}' not supported")

            optimizer.zero_grad()
            loss = sum(losses.values())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()

            tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
            epoch_train_losses.append(loss.item())
            epoch_train_loss_components.append({x: y.item() for x, y in losses.items()})

            if b % 1000 == 0 and b >= 1000:
                time_per_batch = (time.time() - start) / 1000
                print(
                    "\ne{:2d}  b{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(
                        epoch,
                        b,
                        len(dataloader_train),
                        time_per_batch,
                        len(dataloader_train) * time_per_batch / 60,
                    )
                )
                logger.info(
                    "e%d  b%d/%d  %.3fs/batch, %.1fm/epoch",
                    epoch,
                    b,
                    len(dataloader_train),
                    time_per_batch,
                    len(dataloader_train) * time_per_batch / 60,
                )

                mn = pd.concat(tr[-1000:], axis=1).mean(1)
                print(mn)
                logger.info("Loss stats - %s", mn.to_dict())
                start = time.time()

        # Log average train loss for this epoch
        avg_train_loss = np.mean(epoch_train_losses)
        avg_train_loss_components = {
            k: round(
                float(np.mean([d.get(k, 0.0) for d in epoch_train_loss_components])), 4
            )
            for k in epoch_train_loss_components[0].keys()
        }
        logger.info(f"Epoch {epoch} - Average Train Loss: {avg_train_loss:.6f}")
        logger.info(
            f"Epoch {epoch} - Average Train Loss Components: {avg_train_loss_components}"
        )

        logger.info("Evaluating epoch %d", epoch)
        model.eval()
        if conf.dataset == "EASG":
            object_detector_EASG.is_train = False
        else:
            object_detector.is_train = False
        val_losses = []
        val_loss_components = []

        # Initialize predictions collection (only for the best epoch)
        predictions_data = []

        if conf.dataset == "EASG":
            num_top_verb = 5
            num_top_rel_with = 1
            num_top_rel_no = 5
            list_k = [10, 20, 50]
            recall_with = {k: [] for k in list_k}
            recall_no = {k: [] for k in list_k}

        with torch.no_grad():
            test_pbar = tqdm(
                range(len(dataloader_test)), desc=f"Epoch {epoch}/{conf.nepoch} [Eval]"
            )
            for b in test_pbar:
                data = next(test_iter)
                im_data = copy.deepcopy(data[0].cuda(0))
                im_info = copy.deepcopy(data[1].cuda(0))

                # Dataset-specific validation processing
                if conf.dataset == "EASG":
                    gt_grounding = dataset_test.gt_groundings[data[2]]
                    video_list = dataset_test.video_list[data[2]]
                    entry = object_detector_EASG(
                        im_data, im_info, gt_grounding, im_all=None
                    )
                    entry["features_verb"] = copy.deepcopy(
                        dataset_test.verb_feats[data[2]].cuda(0)
                    )
                    pred = model(entry)

                    # EASG-specific evaluation logic (copied from train_with_EASG.py)
                    gcid_to_f_idx = {}
                    for f_idx, fname in enumerate(video_list):
                        gcid = fname.split("_")[0]
                        if gcid not in gcid_to_f_idx:
                            gcid_to_f_idx[gcid] = []
                        gcid_to_f_idx[gcid].append(f_idx)

                    gcid_to_b_idx = {}
                    gcid_obj_idx_to_b_idx = {}
                    b_idx = 0
                    for gcid in gcid_to_f_idx:
                        if gcid not in gcid_to_b_idx:
                            gcid_to_b_idx[gcid] = []

                        if gcid not in gcid_obj_idx_to_b_idx:
                            gcid_obj_idx_to_b_idx[gcid] = {}

                        for f_idx in gcid_to_f_idx[gcid]:
                            for g in gt_grounding[f_idx]:
                                if g["obj"] not in gcid_obj_idx_to_b_idx[gcid]:
                                    gcid_obj_idx_to_b_idx[gcid][g["obj"]] = []

                                gcid_obj_idx_to_b_idx[gcid][g["obj"]].append(b_idx)
                                gcid_to_b_idx[gcid].append(b_idx)
                                b_idx += 1

                    for gcid in gcid_to_f_idx:
                        f_indices = gcid_to_f_idx[gcid]
                        verb_idx = gt_grounding[f_indices[0]][0]["verb"]
                        obj_indices = [g["obj"] for g in gt_grounding[f_indices[0]]]
                        triplets_gt = []
                        for g in gt_grounding[f_indices[0]]:
                            for e in g["edge"]:
                                triplets_gt.append((verb_idx, g["obj"], e))

                        triplets_gt = torch.LongTensor(triplets_gt)

                        num_obj = len(obj_indices)
                        scores_rels = []
                        scores_objs = []
                        for obj_idx in obj_indices:
                            scores_rels.append(
                                pred["edge_distribution"][
                                    gcid_obj_idx_to_b_idx[gcid][obj_idx]
                                ].mean(dim=0)
                            )
                            scores_objs.append(
                                pred["distribution"][
                                    gcid_obj_idx_to_b_idx[gcid][obj_idx]
                                ].mean(dim=0)
                            )

                        scores_rels = torch.stack(scores_rels)
                        scores_objs = torch.stack(scores_objs)
                        scores_verb = pred["distribution_verb"][
                            gcid_to_b_idx[gcid]
                        ].mean(dim=0)

                        triplets_with, triplets_no = get_pred_triplets(
                            conf.mode,
                            verb_idx,
                            obj_indices,
                            scores_rels,
                            scores_objs,
                            scores_verb,
                            list_k,
                            num_top_verb,
                            num_top_rel_with,
                            num_top_rel_no,
                        )

                        out_to_gt_with = intersect_2d(triplets_gt, triplets_with)
                        out_to_gt_no = intersect_2d(triplets_gt, triplets_no)

                        num_gt = triplets_gt.shape[0]
                        for k in list_k:
                            recall_with[k].append(
                                out_to_gt_with[:, :k].any(dim=1).sum().item() / num_gt
                            )
                            recall_no[k].append(
                                out_to_gt_no[:, :k].any(dim=1).sum().item() / num_gt
                            )

                    # EASG validation loss calculation
                    edge_distribution = pred["edge_distribution"]
                    losses = {}
                    if conf.mode != "edgecls":
                        losses["obj_loss"] = ce_loss(
                            pred["distribution"], pred["labels"]
                        )
                    if conf.mode == "easgcls":
                        losses["verb_loss"] = ce_loss(
                            pred["distribution_verb"], pred["labels_verb"]
                        )
                    edge_label = -torch.ones(
                        [len(pred["edge"]), len(dataset_test.edge_classes)],
                        dtype=torch.long,
                    ).to(device=edge_distribution.device)
                    for i in range(len(pred["edge"])):
                        edge_label[i, : len(pred["edge"][i])] = torch.tensor(
                            pred["edge"][i]
                        )
                    losses["edge_loss"] = mlm_loss(edge_distribution, edge_label)

                    # Add validation loss to lists
                    val_loss = sum(losses.values())
                    val_losses.append(val_loss.item())
                    val_loss_components.append({x: y.item() for x, y in losses.items()})

                    # Collect EASG predictions (only for the best epoch)
                    if epoch == best_epoch or epoch == conf.nepoch - 1:
                        for i, video_name in enumerate(video_list):
                            predictions_data.append(
                                {
                                    "dataset": conf.dataset,
                                    "model": conf.model_type,
                                    "mode": conf.mode,
                                    "video": video_name,
                                    "epoch": epoch,
                                    "best_score": best_score,
                                    "best_epoch": best_epoch,
                                }
                            )
                elif conf.dataset == "action_genome":
                    gt_boxes = copy.deepcopy(data[2].cuda(0))
                    num_boxes = copy.deepcopy(data[3].cuda(0))
                    gt_annotation = dataset_test.gt_annotations[data[4]]
                    entry = object_detector(
                        im_data,
                        im_info,
                        gt_boxes,
                        num_boxes,
                        gt_annotation,
                        im_all=None,
                    )

                    if conf.use_matcher:
                        get_sequence(
                            entry,
                            gt_annotation,
                            matcher,
                            (im_info[0][:2] / im_info[0, 2]).cpu().data,
                            conf.mode,
                        )
                    pred = model(entry)

                    # Only compute validation loss if ground truth annotations are available
                    if (
                        "attention_gt" in pred
                        and "spatial_gt" in pred
                        and "contact_gt" in pred
                    ):
                        # Compute validation loss
                        attention_distribution = pred["attention_distribution"]
                        spatial_distribution = pred["spatial_distribution"]
                        contact_distribution = pred["contact_distribution"]
                        attention_label = (
                            torch.tensor(pred["attention_gt"], dtype=torch.long)
                            .to(device=attention_distribution.device)
                            .squeeze()
                        )

                        # Check if batch sizes match before computing loss
                        if attention_distribution.size(0) != attention_label.size(0):
                            logger.warning(
                                f"Batch size mismatch in validation: attention_distribution {attention_distribution.size(0)} vs attention_label {attention_label.size(0)}. Skipping validation loss computation."
                            )
                            continue

                        # Validate that attention labels are within valid range
                        if (
                            attention_label.max() >= attention_distribution.size(1)
                            or attention_label.min() < 0
                        ):
                            logger.warning(
                                f"Invalid attention labels detected: min={attention_label.min()}, max={attention_label.max()}, num_classes={attention_distribution.size(1)}. Skipping validation loss computation."
                            )
                            continue

                        if not conf.bce_loss:
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
                            spatial_label = torch.zeros(
                                [len(pred["spatial_gt"]), 6], dtype=torch.float32
                            ).to(device=attention_distribution.device)
                            contact_label = torch.zeros(
                                [len(pred["contact_gt"]), 17], dtype=torch.float32
                            ).to(device=attention_distribution.device)
                            for i in range(len(pred["spatial_gt"])):
                                spatial_label[i, pred["spatial_gt"][i]] = 1
                                contact_label[i, pred["contact_gt"][i]] = 1

                        # Additional batch size checks for spatial and contact labels
                        if spatial_distribution.size(0) != spatial_label.size(0):
                            logger.warning(
                                f"Batch size mismatch in validation: spatial_distribution {spatial_distribution.size(0)} vs spatial_label {spatial_label.size(0)}. Skipping validation loss computation."
                            )
                            continue

                        if contact_distribution.size(0) != contact_label.size(0):
                            logger.warning(
                                f"Batch size mismatch in validation: contact_distribution {contact_distribution.size(0)} vs contact_label {contact_label.size(0)}. Skipping validation loss computation."
                            )
                            continue

                        try:
                            losses = {}
                            if conf.mode == "sgcls" or conf.mode == "sgdet":
                                # Check object loss batch sizes and validate labels
                                if pred["distribution"].size(0) == pred["labels"].size(
                                    0
                                ):
                                    # Validate object labels are within valid range
                                    if (
                                        pred["labels"].max()
                                        >= pred["distribution"].size(1)
                                        or pred["labels"].min() < 0
                                    ):
                                        logger.warning(
                                            f"Invalid object labels detected: min={pred['labels'].min()}, max={pred['labels'].max()}, num_classes={pred['distribution'].size(1)}. Skipping object loss."
                                        )
                                    else:
                                        losses["object_loss"] = ce_loss(
                                            pred["distribution"], pred["labels"]
                                        )
                                else:
                                    logger.warning(
                                        f"Batch size mismatch in validation: object distribution {pred['distribution'].size(0)} vs labels {pred['labels'].size(0)}. Skipping object loss."
                                    )

                            losses["attention_relation_loss"] = ce_loss(
                                attention_distribution, attention_label
                            )
                            if not conf.bce_loss:
                                losses["spatial_relation_loss"] = mlm_loss(
                                    spatial_distribution, spatial_label
                                )
                                losses["contact_relation_loss"] = mlm_loss(
                                    contact_distribution, contact_label
                                )
                            else:
                                losses["spatial_relation_loss"] = bce_loss(
                                    spatial_distribution, spatial_label
                                )
                                losses["contact_relation_loss"] = bce_loss(
                                    contact_distribution, contact_label
                                )
                            val_loss = sum(losses.values())
                            val_losses.append(val_loss.item())
                            val_loss_components.append(
                                {x: y.item() for x, y in losses.items()}
                            )
                        except RuntimeError as e:
                            if "device-side assert triggered" in str(
                                e
                            ) or "CUDA error" in str(e):
                                logger.warning(
                                    f"CUDA error during validation loss computation: {e}. Skipping this batch."
                                )
                                continue
                            else:
                                raise e

                    # Add evaluator call for Action Genome dataset
                    # Fix missing pred_scores field for TEMPURA and SceneLLM models
                    if (
                        conf.model_type in ["tempura", "scenellm"]
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
                            logger.warning(
                                f"No distribution or labels available to create pred_scores for {conf.model_type}"
                            )
                            continue

                    evaluator.evaluate_scene_graph(gt_annotation, pred)

                    # Collect Action Genome predictions with per-sample metrics (only for the best epoch)
                    if epoch == best_epoch or epoch == conf.nepoch - 1:
                        # Fix missing pred_scores field for TEMPURA and SceneLLM models (for per-sample evaluation)
                        if (
                            conf.model_type in ["tempura", "scenellm"]
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
                                logger.warning(
                                    f"No distribution or labels available to create pred_scores for {conf.model_type} per-sample evaluation"
                                )
                                continue

                        # Get per-sample metrics for this sample
                        per_sample_metrics = evaluator.evaluate_scene_graph(
                            gt_annotation, pred, return_per_sample=True
                        )

                        # Create prediction entry with per-sample metrics
                        pred_entry = {
                            "dataset": conf.dataset,
                            "model": conf.model_type,
                            "mode": conf.mode,
                            "annotation_id": data[4],
                            "epoch": epoch,
                            "best_score": best_score,
                            "best_epoch": best_epoch,
                        }

                        # Add per-sample metrics if available
                        if per_sample_metrics and len(per_sample_metrics) > 0:
                            sample_metrics = per_sample_metrics[
                                0
                            ]  # Take first frame's metrics
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
                else:
                    raise ValueError(f"Dataset '{conf.dataset}' not supported")

        # Log average validation loss for this epoch
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
        logger.info(f"Epoch {epoch} - Average Validation Loss: {avg_val_loss:.6f}")
        logger.info(
            f"Epoch {epoch} - Average Validation Loss Components: {avg_val_loss_components}"
        )

        # Dataset-specific evaluation and scoring
        if conf.dataset == "EASG":
            # Calculate EASG recall metrics
            for k in list_k:
                recall_with[k] = sum(recall_with[k]) / len(recall_with[k]) * 100
                recall_no[k] = sum(recall_no[k]) / len(recall_no[k]) * 100

            msg = "epoch [{:2d}/{:2d}] | With: (".format(epoch, conf.nepoch)
            for k in list_k:
                msg += "{:.1f}, ".format(recall_with[k])
            msg = msg[:-2] + ") | No: ("

            for k in list_k:
                msg += "{:.1f}, ".format(recall_no[k])
            msg = msg[:-2] + ")"

            logger.info(msg)
            logger.info("*" * 40)

            score = (recall_with[20] + recall_no[20]) / 2
        elif conf.dataset == "action_genome":
            # Handle case where evaluator might be empty
            if len(evaluator.result_dict[conf.mode + "_recall"][20]) == 0:
                score = 0.0
                r10 = 0.0
                r20 = 0.0
                r50 = 0.0
                r100 = 0.0
                mrecall = 0.0
                logger.warning("No evaluation results found - using default scores")
            else:
                score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
                r10 = np.mean(evaluator.result_dict[conf.mode + "_recall"][10])
                r20 = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
                r50 = np.mean(evaluator.result_dict[conf.mode + "_recall"][50])
                r100 = np.mean(evaluator.result_dict[conf.mode + "_recall"][100])
                mrecall = evaluator.calc_mrecall()[20]

            evaluator.print_stats()
            evaluator.reset_result()
        else:
            raise ValueError(f"Dataset '{conf.dataset}' not supported")

        scheduler.step(score)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            if conf.model_type == "tempura":
                if epoch > 0 and conf.rel_mem_compute is not None:
                    if len(model.object_classifier.obj_memory) == 0:
                        object_memory = []
                    else:
                        object_memory = model.object_classifier.obj_memory.to("cpu")
                    rel_memory = model.rel_memory
                    if len(rel_memory) != 0:
                        rel_memory = {
                            k: rel_memory[k].to("cpu") for k in rel_memory.keys()
                        }
                else:
                    object_memory = []
                    rel_memory = []
            if not conf.disable_checkpoint_saving:
                torch.save(
                    {"state_dict": model.state_dict()},
                    os.path.join(conf.save_path, "model_best.tar"),
                )
                logger.info("NEW BEST! Saved best checkpoint after %d epochs", epoch)
            else:
                logger.info("NEW BEST! Checkpoint saving disabled (epoch %d)", epoch)

        if mrecall > best_Mrecall:
            best_Mrecall = mrecall
            if not conf.disable_checkpoint_saving:
                torch.save(
                    {"state_dict": model.state_dict()},
                    os.path.join(conf.save_path, "model_best_Mrecall.tar"),
                )
                logger.info(
                    "NEW BEST MRECALL! Saved best checkpoint after %d epochs", epoch
                )
            else:
                logger.info(
                    "NEW BEST MRECALL! Checkpoint saving disabled (epoch %d)", epoch
                )

    logger.info("Training completed!")
    if not conf.disable_checkpoint_saving:
        logger.info(
            "Best model saved at epoch %d with R@20 score: %.4f", best_epoch, best_score
        )
    else:
        logger.info(
            "Best model achieved at epoch %d with R@20 score: %.4f (checkpoint saving disabled)",
            best_epoch,
            best_score,
        )

    # Save final predictions as CSV (only for the best epoch)
    if predictions_data:
        predictions_df = pd.DataFrame(predictions_data)
        predictions_csv_path = os.path.join(conf.save_path, "predictions.csv")
        predictions_df.to_csv(predictions_csv_path, index=False)
        logger.info(f"Predictions saved to: {predictions_csv_path}")
    else:
        logger.warning("No predictions data collected to save")

    if conf.model_type == "tempura" and (conf.rel_mem_compute or conf.obj_mem_compute):
        print("computing memory \n", flush=True)
        rel_class_num = {
            "attention": model.attention_class_num,
            "spatial": model.spatial_class_num,
            "contacting": model.contact_class_num,
        }
        if conf.tracking:
            obj_feature_dim = 2048 + 200 + 128
        else:
            obj_feature_dim = 1024
        rel_memory, obj_memory = memory_computation(
            unc_vals,
            conf.save_path,
            rel_class_num,
            len(model.obj_classes),
            obj_feature_dim=obj_feature_dim,
            rel_feature_dim=1936,
            obj_weight_type=conf.obj_mem_weight_type,
            rel_weight_type=conf.rel_mem_weight_type,
            obj_mem=conf.obj_mem_compute,
            obj_unc=conf.obj_unc,
            include_bg_mem=False,
        )

        model.object_classifier.obj_memory = obj_memory.to(gpu_device)
        model.rel_memory = {k: rel_memory[k].to(gpu_device) for k in rel_memory.keys()}
