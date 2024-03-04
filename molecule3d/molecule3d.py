import datasets
from datasets import DatasetBuilder, BuilderConfig, GeneratorBasedBuilder
from datasets.download.download_manager import DownloadManager
from datasets.info import DatasetInfo
import os.path as osp
import pandas as pd
from rdkit import Chem
from .utils import mol_to_graph_dict
from typing import Literal


_ROOT_URL = "https://huggingface.co/datasets/RichXuOvO/HFMolecule3D/resolve/main"

_RANDOM_URLS = {
    "train": [
        f"{_ROOT_URL}/random_train_0.sdf",
        f"{_ROOT_URL}/random_train_1.sdf",
        f"{_ROOT_URL}/random_train_2.sdf",
        f"{_ROOT_URL}/random_train_3.sdf",
        f"{_ROOT_URL}/random_train.csv",
    ],
    "valid": [
        f"{_ROOT_URL}/random_valid.sdf",
        f"{_ROOT_URL}/random_valid.csv",
    ],
    "test": [
        f"{_ROOT_URL}/random_test.sdf",
        f"{_ROOT_URL}/random_test.csv",
    ],
}

_SCAFFOLD_URLS = {
    "train": [
        f"{_ROOT_URL}/scaffold_train_0.sdf",
        f"{_ROOT_URL}/scaffold_train_1.sdf",
        f"{_ROOT_URL}/scaffold_train_2.sdf",
        f"{_ROOT_URL}/scaffold_train_3.sdf",
        f"{_ROOT_URL}/scaffold_train.csv",
    ],
    "valid": [
        f"{_ROOT_URL}/scaffold_valid.sdf",
        f"{_ROOT_URL}/scaffold_valid.csv",
    ],
    "test": [
        f"{_ROOT_URL}/scaffold_test.sdf",
        f"{_ROOT_URL}/scaffold_test.csv",
    ],
}


class HFMolecule3DRandomSplit(GeneratorBasedBuilder):
    """Random Split of Molecule3D dataset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="HFMolecule3DRandomSplit",
            version=VERSION,
            description="Random Split of Molecule3D dataset.",
        )
    ]

    def __init__(self, **kwargs):
        super(HFMolecule3DRandomSplit, self).__init__(**kwargs)

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description="Random Split of Molecule3D dataset.",
        )

    def _split_generators(self, dl_manager: DownloadManager):
        archives = dl_manager.download(_RANDOM_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "archives": archives["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "archives": archives["valid"],
                    "split": "valid",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "archives": archives["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, archives, split):
        """Yields examples."""
        if split == "train":
            sdf_files, csv_file = archives[:-1], archives[-1]
            suppliers = [Chem.SDMolSupplier(fileName=file, removeHs=False, sanitize=True) for file in sdf_files]
            properties_df = pd.read_csv(csv_file)
            for idx, row_series in properties_df.iterrows():
                row, col = idx // 600000, idx % 600000
                mol = suppliers[row][col]
                yield idx, mol_to_graph_dict(molecule=mol, properties_dict=row_series.to_dict())
        elif split == "valid" or split == "test":
            sdf_file, csv_file = archives[0], archives[1]
            suppliers = Chem.SDMolSupplier(fileName=sdf_file, removeHs=False, sanitize=True)
            properties_df = pd.read_csv(csv_file)
            for idx, row_series in properties_df.iterrows():
                mol = suppliers[idx]
                yield idx, mol_to_graph_dict(molecule=mol, properties_dict=row_series.to_dict())


class HFMolecule3DScaffoldSplit(GeneratorBasedBuilder):
    """Scaffold Split of Molecule3D dataset."""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        BuilderConfig(
            name="HFMolecule3DScaffoldSplit",
            version=VERSION,
            description="Scaffold Split of Molecule3D dataset.",
        )
    ]

    def __init__(self, **kwargs):
        super(HFMolecule3DScaffoldSplit, self).__init__(**kwargs)

    def _info(self) -> DatasetInfo:
        return datasets.DatasetInfo(
            description="Scaffold Split of Molecule3D dataset.",
        )

    def _split_generators(self, dl_manager: DownloadManager):
        archives = dl_manager.download(_SCAFFOLD_URLS)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "archives": archives["train"],
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "archives": archives["valid"],
                    "split": "valid",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "archives": archives["test"],
                    "split": "test",
                },
            ),
        ]

    def _generate_examples(self, archives, split):
        """Yields examples."""
        if split == "train":
            sdf_files, csv_file = archives[:-1], archives[-1]
            suppliers = [Chem.SDMolSupplier(fileName=file, removeHs=False, sanitize=True) for file in sdf_files]
            properties_df = pd.read_csv(csv_file)
            for idx, row_series in properties_df.iterrows():
                row, col = idx // 600000, idx % 600000
                mol = suppliers[row][col]
                yield idx, mol_to_graph_dict(molecule=mol, properties_dict=row_series.to_dict())
        elif split == "valid" or split == "test":
            sdf_file, csv_file = archives[0], archives[1]
            suppliers = Chem.SDMolSupplier(fileName=sdf_file, removeHs=False, sanitize=True)
            properties_df = pd.read_csv(csv_file)
            for idx, row_series in properties_df.iterrows():
                mol = suppliers[idx]
                yield idx, mol_to_graph_dict(molecule=mol, properties_dict=row_series.to_dict())


if __name__ == "__main__":
    # dataset = HFMolecule3DRandomSplit(use_auth_token=True)
    dataset = HFMolecule3DScaffoldSplit(use_auth_token=True)
    dataset.download_and_prepare()
    dataset = dataset.as_dataset()
    print(dataset['train'][2])
    print(dataset)
