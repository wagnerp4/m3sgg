from math import ceil

import torch


def intersect_2d(out, gt):
    return (out[..., None] == gt.T[None, ...]).all(1)


def get_pred_triplets(
    mode,
    verb_idx,
    obj_indices,
    scores_rels,
    scores_objs,
    scores_verb,
    list_k,
    num_top_verb,
    num_top_rel_with,
    num_top_rel_no,
):
    triplets_with, triplets_no = [], []
    scores_with, scores_no = [], []
    num_obj = len(obj_indices)
    if mode == "edgecls":  # alias for predcls
        for obj_idx, scores_rel in zip(obj_indices, scores_rels):
            sorted_scores_rel = scores_rel.argsort(descending=True)
            for ri in sorted_scores_rel[:num_top_rel_with]:
                triplets_with.append((verb_idx, obj_idx, ri.item()))
                scores_with.append(scores_rel[ri].item())

            for ri in sorted_scores_rel[: ceil(max(list_k) / num_obj)]:
                triplets_no.append((verb_idx, obj_idx, ri.item()))
                scores_no.append(scores_rel[ri].item())
    elif mode == "sgcls":
        num_top_obj_with = ceil(max(list_k) / (num_top_rel_with * num_obj))
        num_top_obj_no = ceil(max(list_k) / (num_top_rel_no * num_obj))
        for scores_obj, scores_rel in zip(scores_objs, scores_rels):
            sorted_scores_obj = scores_obj.argsort(descending=True)
            sorted_scores_rel = scores_rel.argsort(descending=True)
            for oi in sorted_scores_obj[:num_top_obj_with]:
                for ri in sorted_scores_rel[:num_top_rel_with]:
                    triplets_with.append((verb_idx, oi.item(), ri.item()))
                    scores_with.append((scores_obj[oi] + scores_rel[ri]).item())
            for oi in sorted_scores_obj[:num_top_obj_no]:
                for ri in sorted_scores_rel[:num_top_rel_no]:
                    triplets_no.append((verb_idx, oi.item(), ri.item()))
                    scores_no.append((scores_obj[oi] + scores_rel[ri]).item())
    elif mode == "easgcls":
        num_top_obj_with = ceil(
            max(list_k) / (num_top_verb * num_top_rel_with * num_obj)
        )
        num_top_obj_no = ceil(max(list_k) / (num_top_verb * num_top_rel_no * num_obj))
        for vi in scores_verb.argsort(descending=True)[:num_top_verb]:
            for scores_obj, scores_rel in zip(scores_objs, scores_rels):
                sorted_scores_obj = scores_obj.argsort(descending=True)
                sorted_scores_rel = scores_rel.argsort(descending=True)
                for oi in sorted_scores_obj[:num_top_obj_with]:
                    for ri in sorted_scores_rel[:num_top_rel_with]:
                        triplets_with.append((vi.item(), oi.item(), ri.item()))
                        scores_with.append(
                            (scores_verb[vi] + scores_obj[oi] + scores_rel[ri]).item()
                        )
                for oi in sorted_scores_obj[:num_top_obj_no]:
                    for ri in sorted_scores_rel[:num_top_rel_no]:
                        triplets_no.append((vi.item(), oi.item(), ri.item()))
                        scores_no.append(
                            (scores_verb[vi] + scores_obj[oi] + scores_rel[ri]).item()
                        )

    triplets_with = torch.tensor(triplets_with, dtype=torch.long)
    triplets_no = torch.tensor(triplets_no, dtype=torch.long)

    triplets_with = triplets_with[
        torch.argsort(torch.tensor(scores_with), descending=True)
    ]
    triplets_no = triplets_no[torch.argsort(torch.tensor(scores_no), descending=True)]

    return triplets_with, triplets_no


def create_subset_samplers(train_size, test_size, fraction=1, seed=42):
    """
    Create subset samplers for training and testing datasets.

    Args:
        train_size (int): Size of training dataset
        test_size (int): Size of test dataset
        fraction (int): Fraction of the dataset to use (1 = use all, 2 = use half, etc.)
        seed (int): Random seed for reproducibility

    Returns:
        tuple: (train_sampler, test_sampler, train_subset_size, test_subset_size)
    """
    train_subset_size = train_size // fraction
    test_subset_size = test_size // fraction

    torch.manual_seed(seed)
    train_indices = torch.randperm(train_size)[:train_subset_size]
    test_indices = torch.randperm(test_size)[:test_subset_size]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    test_sampler = torch.utils.data.SubsetRandomSampler(test_indices)

    return train_sampler, test_sampler, train_subset_size, test_subset_size


def create_dataloaders(
    dataset_train,
    dataset_test,
    train_sampler,
    test_sampler,
    collate_fn,
    num_workers=0,
    pin_memory=False,
):
    """Create training and test DataLoaders with consistent settings.

    :param dataset_train: Training dataset
    :type dataset_train: torch.utils.data.Dataset
    :param dataset_test: Test dataset
    :type dataset_test: torch.utils.data.Dataset
    :param train_sampler: Training sampler
    :type train_sampler: torch.utils.data.Sampler
    :param test_sampler: Test sampler
    :type test_sampler: torch.utils.data.Sampler
    :param collate_fn: Collate function for batching
    :type collate_fn: callable
    :param num_workers: Number of DataLoader workers
    :type num_workers: int
    :param pin_memory: Whether to pin memory
    :type pin_memory: bool
    :return: Tuple of (train_dataloader, test_dataloader)
    :rtype: tuple
    """
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=test_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    return dataloader_train, dataloader_test
