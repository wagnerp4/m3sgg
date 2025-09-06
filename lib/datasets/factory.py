from typing import Any, Tuple

from lib.datasets.action_genome import AG
from lib.datasets.easg import EASG


def get_datasets(conf: Any) -> Tuple[object, object]:
    """Return initialized training and test dataset instances for the selected dataset.

    This selects the appropriate dataset class and its constructor arguments based
    on ``conf.dataset`` and related configuration flags.

    :param conf: The experiment configuration object containing dataset settings
    :type conf: Any
    :return: A tuple of (dataset_train, dataset_test) instances
    :rtype: tuple
    """
    registry = {
        "EASG": lambda c: (
            EASG(split="train", datasize=c.datasize, data_path=c.data_path),
            EASG(split="val", datasize=c.datasize, data_path=c.data_path),
        ),
        "action_genome": lambda c: (
            AG(
                mode="train",
                datasize=c.datasize,
                data_path=c.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if c.mode == "predcls" else True,
            ),
            AG(
                mode="test",
                datasize=c.datasize,
                data_path=c.data_path,
                filter_nonperson_box_frame=True,
                filter_small_box=False if c.mode == "predcls" else True,
            ),
        ),
    }
    if conf.dataset not in registry:
        raise ValueError(f"Dataset '{conf.dataset}' not supported")
    return registry[conf.dataset](conf)
