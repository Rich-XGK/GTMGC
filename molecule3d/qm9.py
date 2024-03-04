import torch
import datasets
from datasets import DatasetBuilder, BuilderConfig, GeneratorBasedBuilder
from datasets.download.download_manager import DownloadManager
from datasets.info import DatasetInfo
import os.path as osp
import pandas as pd
from rdkit import Chem
from .utils import mol_to_graph_dict
from typing import Literal

HAR2EV = 27.211386246  # Hartree to eV
KCALMOL2EV = 0.04336414  # kcal/mol to eV

CONVERSION = {
    "A": 1.0,
    "B": 1.0,
    "C": 1.0,
    "mu": 1.0,
    "alpha": 1.0,
    "homo": HAR2EV,
    "lumo": HAR2EV,
    "gap": HAR2EV,
    "r2": 1.0,
    "zpve": HAR2EV,
    "u0": HAR2EV,
    "u298": HAR2EV,
    "h298": HAR2EV,
    "g298": HAR2EV,
    "cv": 1.0,
    "u0_atom": KCALMOL2EV,
    "u298_atom": KCALMOL2EV,
    "h298_atom": KCALMOL2EV,
    "g298_atom": KCALMOL2EV,
}


_ROOT_URL = "https://huggingface.co/datasets/RichXuOvO/HFQm9/resolve/main"

_URLS = {
    "raw_sdf": f"{_ROOT_URL}/gdb9.sdf",
    "properties_csv": f"{_ROOT_URL}/gdb9.sdf.csv",
    "train_csv": f"{_ROOT_URL}/train_indices.csv",
    "valid_csv": f"{_ROOT_URL}/valid_indices.csv",
    "test_csv": f"{_ROOT_URL}/test_indices.csv",
}


class HFQm9(GeneratorBasedBuilder):
    """ "QM9 dataset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="HFQm9",
            version=VERSION,
            description="QM9 dataset.",
        )
    ]

    def __init__(self, standardize: bool = True, **kwargs):
        """Initialize HFQm9 dataset builder.

        Args:
            standardize (bool, optional): followed https://arxiv.org/abs/2206.11990,
            normalized all properties by subtracting the mean and dividing by the Mean Absolute Deviation.
        """
        self.standardize = standardize
        super(HFQm9, self).__init__(**kwargs)

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description="QM9 dataset.",
        )

    def _split_generators(self, dl_manager: DownloadManager):
        archives = dl_manager.download(_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "raw_sdf_path": archives["raw_sdf"],
                    "properties_csv_path": archives["properties_csv"],
                    "indices_csv_path": archives["train_csv"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "raw_sdf_path": archives["raw_sdf"],
                    "properties_csv_path": archives["properties_csv"],
                    "indices_csv_path": archives["valid_csv"],
                    "split": "valid",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "raw_sdf_path": archives["raw_sdf"],
                    "properties_csv_path": archives["properties_csv"],
                    "indices_csv_path": archives["test_csv"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, raw_sdf_path, properties_csv_path, indices_csv_path, split):
        properties_df = pd.read_csv(properties_csv_path)
        for column in CONVERSION.keys():  # convert units
            properties_df[column] = properties_df[column] * CONVERSION[column]
        columns = properties_df.drop(columns="mol_id").columns
        if self.standardize:
            for col in columns:  # standardize properties by subtracting the mean and dividing by the Mean Absolute Deviation
                if col in ["u0_atom", "u298_atom", "h298_atom", "g298_atom"]:
                    mean, std = 0, 1
                else:
                    mean = properties_df[col].mean()
                    std = properties_df[col].std()
                properties_df[col] = (properties_df[col] - mean) / std
        indices_df = pd.read_csv(indices_csv_path)
        indices = indices_df["index"].values.tolist()
        supplier = Chem.SDMolSupplier(fileName=raw_sdf_path, removeHs=False, sanitize=True)
        for idx in indices:
            mol = supplier[idx]
            properties_dict = properties_df.iloc[idx].to_dict()
            if mol is None:
                continue
            graph_dict = mol_to_graph_dict(mol, properties_dict)
            yield idx, graph_dict


if __name__ == "__main__":
    dataset = HFQm9(use_auth_token=True)
    dataset = dataset.download_and_prepare()
    dataset = dataset.as_dataset()
    print(dataset)
