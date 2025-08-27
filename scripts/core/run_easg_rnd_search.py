import logging
import os
import pickle
import random
from argparse import ArgumentParser
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import (
    BCEWithLogitsLoss,
    CrossEntropyLoss,
    Linear,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)

from run_easg import EASGData, evaluation


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("path_to_annotations", type=Path)
    parser.add_argument("path_to_data", type=Path)
    parser.add_argument("path_to_output", type=Path)
    parser.add_argument("--gpu_id", type=int, default=0, help="which gpu to run")
    parser.add_argument(
        "--num_trials", type=int, default=10, help="number of random search trials"
    )
    parser.add_argument("--epochs", type=int, default=30, help="epochs per trial")
    parser.add_argument(
        "--random_guess", action="store_true", help="for random guessing"
    )
    return parser.parse_args()


def make_easg_model(
    verbs,
    objs,
    rels,
    dim_clip_feat,
    dim_obj_feat,
    dropout_p,
    activation,
    n_linear,
    residual,
):
    class CustomEASG(Module):
        def __init__(self):
            super().__init__()
            act = {"relu": ReLU(), "sigmoid": Sigmoid(), "tanh": Tanh()}[activation]
            # Verb branch
            verb_layers = []
            in_dim = dim_clip_feat
            for i in range(n_linear):
                out_dim = 1024 if i < n_linear - 1 else len(verbs)
                verb_layers.append(Linear(in_dim, out_dim))
                if i < n_linear - 1:
                    verb_layers.append(act)
                in_dim = out_dim
            self.verb_branch = Sequential(*verb_layers)
            # Obj branch
            obj_layers = []
            in_dim = dim_obj_feat
            for i in range(n_linear):
                out_dim = 512 if i < n_linear - 1 else len(objs)
                obj_layers.append(Linear(in_dim, out_dim))
                if i < n_linear - 1:
                    obj_layers.append(act)
                in_dim = out_dim
            self.obj_branch = Sequential(*obj_layers)
            # Rel branch
            rel_layers = []
            in_dim = dim_clip_feat + dim_obj_feat
            for i in range(n_linear):
                out_dim = 1024 if i < n_linear - 1 else len(rels)
                rel_layers.append(Linear(in_dim, out_dim))
                if i < n_linear - 1:
                    rel_layers.append(act)
                in_dim = out_dim
            self.rel_branch = Sequential(*rel_layers)
            self.residual = residual
            self.n_linear = n_linear
            self.act = act

        def forward(self, clip_feat, obj_feats):
            out_verb = self.verb_branch(clip_feat)
            out_objs = self.obj_branch(obj_feats)
            clip_feat_expanded = clip_feat.expand(obj_feats.shape[0], -1)
            rel_input = torch.cat((clip_feat_expanded, obj_feats), dim=1)
            out_rels = self.rel_branch(rel_input)
            # Optionally add residual connections
            if self.residual and self.n_linear > 1:
                out_verb = out_verb + clip_feat[..., : out_verb.shape[-1]]
                out_objs = out_objs + obj_feats[..., : out_objs.shape[-1]]
                out_rels = out_rels + rel_input[..., : out_rels.shape[-1]]
            return out_verb, out_objs, out_rels

    return CustomEASG()


def random_search_architecture(
    args,
    verbs,
    objs,
    rels,
    dataset_train,
    dataset_val,
    search_space,
    num_trials=10,
    epochs=30,
):
    best_result = None
    best_config = None
    results = []
    for trial in range(num_trials):
        config = {k: random.choice(v) for k, v in search_space.items()}
        print(f"\n[RandomSearch] Trial {trial+1}/{num_trials} Config: {config}")
        model = make_easg_model(
            verbs,
            objs,
            rels,
            dim_clip_feat=config["dim_clip_feat"],
            dim_obj_feat=config["dim_obj_feat"],
            dropout_p=config["dropout_p"],
            activation=config["activation"],
            n_linear=config["n_linear"],
            residual=config["residual"],
        )
        device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() else "cpu"
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        criterion = CrossEntropyLoss()
        criterion_rel = BCEWithLogitsLoss()
        batch_size = config["batch_size"]
        for epoch in range(1, epochs + 1):
            model.train()
            list_index = list(range(len(dataset_train)))
            random.shuffle(list_index)
            for i in range(0, len(list_index), batch_size):
                batch_indices = list_index[i : i + batch_size]
                optimizer.zero_grad()
                loss = 0
                for idx in batch_indices:
                    graph = dataset_train[idx]
                    clip_feat = graph["clip_feat"].unsqueeze(0).to(device)
                    obj_feats = graph["obj_feats"].to(device)
                    out_verb, out_objs, out_rels = model(clip_feat, obj_feats)
                    verb_idx = graph["verb_idx"].to(device)
                    obj_indices = graph["obj_indices"].to(device)
                    rels_vecs = graph["rels_vecs"].to(device)
                    loss = loss + (
                        criterion(out_verb, verb_idx)
                        + criterion(out_objs, obj_indices)
                        + criterion_rel(out_rels, rels_vecs)
                    )
                loss = loss / len(batch_indices)
                loss.backward()
                optimizer.step()
            scheduler.step()
        (
            val_loss,
            recall_predcls_with,
            recall_predcls_no,
            recall_sgcls_with,
            recall_sgcls_no,
            recall_easgcls_with,
            recall_easgcls_no,
        ) = evaluation(dataset_val, model, device, args)
        result = {
            "config": config,
            "recall_predcls_with": recall_predcls_with,
            "recall_sgcls_with": recall_sgcls_with,
            "recall_easgcls_with": recall_easgcls_with,
            "recall_predcls_no": recall_predcls_no,
            "recall_sgcls_no": recall_sgcls_no,
            "recall_easgcls_no": recall_easgcls_no,
            "val_loss": val_loss,
        }
        results.append(result)
        print(
            f"[RandomSearch] Trial {trial+1} Results: easgcls@20={recall_easgcls_with[20]:.2f}, sgcls@20={recall_sgcls_with[20]:.2f}, predcls@20={recall_predcls_with[20]:.2f}"
        )
        if (
            best_result is None
            or recall_easgcls_with[20] > best_result["recall_easgcls_with"][20]
        ):
            best_result = result
            best_config = config
    print("\n[RandomSearch] Best Config:", best_config)
    print("[RandomSearch] Best Results:", best_result)
    with open(args.path_to_output / "random_search_results.pkl", "wb") as f:
        pickle.dump(results, f)
    return best_config, best_result


def main():
    args = parse_args()
    args.path_to_output.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        format="%(asctime)s.%(msecs)03d %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(filename=args.path_to_output / "log", mode="w"),
        ],
    )
    logger = logging.getLogger()
    with open(args.path_to_annotations / "verbs.txt") as f:
        verbs = [l.strip() for l in f.readlines()]
    with open(args.path_to_annotations / "objects.txt") as f:
        objs = [l.strip() for l in f.readlines()]
    with open(args.path_to_annotations / "relationships.txt") as f:
        rels = [l.strip() for l in f.readlines()]
    dataset_train = EASGData(
        args.path_to_annotations, args.path_to_data, "train", verbs, objs, rels
    )
    dataset_val = EASGData(
        args.path_to_annotations, args.path_to_data, "val", verbs, objs, rels
    )
    search_space = {
        "dim_clip_feat": [2304],
        "dim_obj_feat": [1024],
        "dropout_p": [0.1, 0.3, 0.5],
        "activation": ["relu", "sigmoid", "tanh"],
        "n_linear": [1, 2, 3],
        "residual": [True, False],
        "batch_size": [1, 2, 4, 8],
        "lr": [1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
    }
    best_config, best_result = random_search_architecture(
        args,
        verbs,
        objs,
        rels,
        dataset_train,
        dataset_val,
        search_space,
        num_trials=args.num_trials,
        epochs=args.epochs,
    )
    print("\nBest architecture config:", best_config)
    print("Best recall values:", best_result)


if __name__ == "__main__":
    main()
