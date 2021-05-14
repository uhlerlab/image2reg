from typing import Iterable, List

import numpy as np
import torch
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.data.datasets import LabeledDataset, TorchTransformableSubset


class BaseDataHandler(object):
    def __init__(
        self,
        dataset: LabeledDataset,
        batch_size: int = 64,
        num_workers: int = 0,
        transformation_dict: dict = None,
        random_state: int = 42,
        drop_last_batch: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformation_dict = transformation_dict
        self.random_state = random_state
        self.drop_last_batch = drop_last_batch


class DataHandler(BaseDataHandler):
    def __init__(
        self,
        dataset: LabeledDataset,
        batch_size: int = 64,
        num_workers: int = 0,
        transformation_dict: dict = None,
        random_state: int = 42,
        drop_last_batch: bool = True,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            transformation_dict=transformation_dict,
            random_state=random_state,
            drop_last_batch=drop_last_batch,
        )
        self.train_val_test_datasets_dict = None
        self.data_loader_dict = None

    def stratified_train_val_test_split(self, splits: Iterable) -> None:
        indices = np.array(list(range(len(self.dataset))))
        labels = np.array(self.dataset.labels)
        train_portion, val_portion, test_portion = splits[0], splits[1], splits[2]

        train_and_val_idc, test_idc = train_test_split(
            indices,
            test_size=test_portion,
            stratify=labels,
            random_state=self.random_state,
        )

        train_idc, val_idc = train_test_split(
            train_and_val_idc,
            test_size=val_portion / (1 - test_portion),
            stratify=labels[train_and_val_idc],
            random_state=self.random_state,
        )

        train_dataset = TorchTransformableSubset(
            dataset=self.dataset, indices=train_idc
        )
        val_dataset = TorchTransformableSubset(dataset=self.dataset, indices=val_idc)
        test_dataset = TorchTransformableSubset(dataset=self.dataset, indices=test_idc)

        self.train_val_test_datasets_dict = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }

    def get_data_loader_dict(self, shuffle: bool = True,) -> None:
        if self.transformation_dict is not None:
            for k, transform_pipeline in self.transformation_dict.items():
                self.train_val_test_datasets_dict[k].set_transform_pipeline(
                    transform_pipeline
                )
        data_loader_dict = {}
        for k, dataset in self.train_val_test_datasets_dict.items():
            data_loader_dict[k] = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=shuffle and k == "train",
                num_workers=self.num_workers,
                drop_last=self.drop_last_batch,
            )

        self.data_loader_dict = data_loader_dict


class DataHandlerCV(BaseDataHandler):
    def __init__(
        self,
        dataset: LabeledDataset,
        n_folds: int = 4,
        train_val_split: List = [0.8, 0.2],
        batch_size: int = 64,
        num_workers: int = 0,
        transformation_dict: dict = None,
        random_state: int = 42,
        drop_last_batch: bool = True,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            transformation_dict=transformation_dict,
            random_state=random_state,
            drop_last_batch=drop_last_batch,
        )

        self.n_folds = n_folds
        self.train_val_split = train_val_split

        # The training dataset will be all folds except the one left out for validation split into train and validation
        # portion to trigger e.g. early stopping and select the best performing model
        # The test dataset will be the hold-out fold.
        self.train_val_test_datasets = None
        self.data_loader_dicts = None

    def stratified_kfold_split(self):
        splitter = StratifiedKFold(
            n_splits=self.n_folds, shuffle=False, random_state=self.random_state
        )
        indices = np.array(list(range(len(self.dataset))))
        labels = np.array(self.dataset.labels)

        if self.train_val_test_datasets is None:
            self.train_val_test_datasets = []

        for train_fold_idc, test_fold_idc in splitter.split(X=indices, y=labels):
            test_dataset = TorchTransformableSubset(
                dataset=self.dataset, indices=test_fold_idc
            )

            # Split train_fold into train and validation set
            train_idc, val_idc = train_test_split(
                train_fold_idc,
                test_size=self.train_val_split[1],
                stratify=labels[train_fold_idc],
                random_state=self.random_state,
            )
            train_dataset = TorchTransformableSubset(
                dataset=self.dataset, indices=train_idc
            )
            val_dataset = TorchTransformableSubset(
                dataset=self.dataset, indices=val_idc
            )

            train_val_test_dataset_dict = {
                "train": train_dataset,
                "val": val_dataset,
                "test": test_dataset,
            }
            self.train_val_test_datasets.append(train_val_test_dataset_dict)

    def get_data_loader_dicts(self, shuffle: bool = True):

        if self.data_loader_dicts is None:
            self.data_loader_dicts = []
        for train_val_test_datasets_dict in self.train_val_test_datasets:
            if self.transformation_dict is not None:
                for k, transform_pipeline in self.transformation_dict.items():
                    train_val_test_datasets_dict[k].set_transform_pipeline(
                        transform_pipeline
                    )
            data_loader_dict = {}
            for k, dataset in train_val_test_datasets_dict.items():
                if shuffle and k == "train":
                    generator = torch.Generator().manual_seed(self.random_state)
                    sampler = RandomSampler(data_source=dataset, generator=generator)
                else:
                    sampler = SequentialSampler(data_source=dataset)
                data_loader_dict[k] = DataLoader(
                    dataset=dataset,
                    batch_size=self.batch_size,
                    sampler=sampler,
                    num_workers=self.num_workers,
                    drop_last=self.drop_last_batch,
                )

            self.data_loader_dicts.append(data_loader_dict)
