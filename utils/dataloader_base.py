from abc import abstractmethod
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from utils.config_loading import instantiate_from_config


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class BaseDataset(IterableDataset):
    """
    Define an interface to make the IterableDatasets for text2img data chainable
    """

    def __init__(self, num_records=0, valid_ids=None, size=256):
        super().__init__()
        self.num_records = num_records
        self.valid_ids = valid_ids
        self.sample_ids = valid_ids
        self.size = size

        print(f"{self.__class__.__name__} dataset contains {self.__len__()} examples.")

    def __len__(self):
        return self.num_records

    @abstractmethod
    def __iter__(self):
        pass


def worker_init_fn(_):
    worker_info = torch.utils.data.get_worker_info()  # type: ignore

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, BaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size : (worker_id + 1) * split_size]  # type: ignore
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)  # type: ignore
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)  # type: ignore
    return np.random.seed(np.random.get_state()[1][0] + worker_id)  # type: ignore


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train: BaseDataset | None = None,
        validation: BaseDataset | None = None,
        test: BaseDataset | None = None,
        predict: BaseDataset | None = None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = {}
        self.num_workers = num_workers if num_workers is not None else (batch_size * 2)
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.train = train
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.validation = validation
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.test = test
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.predict = predict
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):  # noqa: ARG002
        self.datasets = {k: instantiate_from_config(self.dataset_configs[k]) for k in self.dataset_configs}
        for k, i in self.datasets.items():
            setattr(self, k, i)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets["train"], BaseDataset)
        init_fn = worker_init_fn if is_iterable_dataset or self.use_worker_init_fn else None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size, num_workers=self.num_workers, shuffle=not is_iterable_dataset, worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        init_fn = worker_init_fn if isinstance(self.datasets["validation"], BaseDataset) or self.use_worker_init_fn else None
        return DataLoader(self.datasets["validation"], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets["train"], BaseDataset)
        init_fn = worker_init_fn if is_iterable_dataset or self.use_worker_init_fn else None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, _shuffle=False):
        init_fn = worker_init_fn if isinstance(self.datasets["predict"], BaseDataset) or self.use_worker_init_fn else None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size, num_workers=self.num_workers, worker_init_fn=init_fn)


def get_dataset(config) -> DataModuleFromConfig:
    data: DataModuleFromConfig = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    for k in data.datasets:
        print(f"{k}, {type(data.datasets[k])}, {len(data.datasets[k])}")
    return data
