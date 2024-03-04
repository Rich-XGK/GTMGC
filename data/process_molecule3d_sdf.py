import os
import gc
import argparse
import pandas as pd
import numpy as np
import os.path as osp
import rdkit.Chem as Chem
from tqdm import tqdm


def process(args: argparse.Namespace):
    # raw files
    sdf_files = [
        "combined_mols_0_to_1000000.sdf",
        "combined_mols_1000000_to_2000000.sdf",
        "combined_mols_2000000_to_3000000.sdf",
        "combined_mols_3000000_to_3899647.sdf",
    ]
    random_split_files = ["random_train.csv", "random_valid.csv", "random_test.csv"]
    scaffold_split_files = ["scaffold_train.csv", "scaffold_valid.csv", "scaffold_test.csv"]
    properties_file = "properties.csv"
    raw_files = set(os.listdir(args.data_dir))
    assert set(sdf_files).issubset(raw_files), "The sdf files are not complete."
    assert properties_file in raw_files, "The properties file is not complete."
    assert set(random_split_files).issubset(raw_files), "The random split files are not complete."
    assert set(scaffold_split_files).issubset(raw_files), "The scaffold split files are not complete."
    sdf_files = [osp.join(args.data_dir, file) for file in sdf_files]
    properties_file = osp.join(args.data_dir, properties_file)
    random_split_files = [osp.join(args.data_dir, file) for file in random_split_files]
    scaffold_split_files = [osp.join(args.data_dir, file) for file in scaffold_split_files]
    # removeHs=False, sanitize=True similar as the original code in Molecule3D
    supplier_ls = [Chem.SDMolSupplier(fileName=file, removeHs=False, sanitize=True) for file in sdf_files]
    properties_df = pd.read_csv(properties_file)
    random_train_indices = pd.read_csv(random_split_files[0]).values
    random_valid_indices = pd.read_csv(random_split_files[1]).values
    random_test_indices = pd.read_csv(random_split_files[2]).values
    scaffold_train_indices = pd.read_csv(scaffold_split_files[0]).values
    scaffold_valid_indices = pd.read_csv(scaffold_split_files[1]).values
    scaffold_test_indices = pd.read_csv(scaffold_split_files[2]).values
    # Collecting properties
    print("Collecting properties...")
    if args.random_train:
        random_train_properties = properties_df.iloc[random_train_indices[:, 0]]
        random_train_properties.to_csv(osp.join(args.out_dir, "random/train.csv"), index=False)
    if args.random_valid:
        random_valid_properties = properties_df.iloc[random_valid_indices[:, 0]]
        random_valid_properties.to_csv(osp.join(args.out_dir, "random/valid.csv"), index=False)
    if args.random_test:
        random_test_properties = properties_df.iloc[random_test_indices[:, 0]]
        random_test_properties.to_csv(osp.join(args.out_dir, "random/test.csv"), index=False)
    if args.scaffold_train:
        scaffold_train_properties = properties_df.iloc[scaffold_train_indices[:, 0]]
        scaffold_train_properties.to_csv(osp.join(args.out_dir, "scaffold/train.csv"), index=False)
    if args.scaffold_valid:
        scaffold_valid_properties = properties_df.iloc[scaffold_valid_indices[:, 0]]
        scaffold_valid_properties.to_csv(osp.join(args.out_dir, "scaffold/valid.csv"), index=False)
    if args.scaffold_test:
        scaffold_test_properties = properties_df.iloc[scaffold_test_indices[:, 0]]
        scaffold_test_properties.to_csv(osp.join(args.out_dir, "scaffold/test.csv"), index=False)
    print("Collecting properties done.")
    # Collecting graphs
    print("Collecting graphs...")
    # convert absolute indices to relative indices(row, col)
    random_train_indices = np.hstack((random_train_indices // 1000000, random_train_indices % 1000000)).tolist()
    random_valid_indices = np.hstack((random_valid_indices // 1000000, random_valid_indices % 1000000)).tolist()
    random_test_indices = np.hstack((random_test_indices // 1000000, random_test_indices % 1000000)).tolist()
    scaffold_train_indices = np.hstack((scaffold_train_indices // 1000000, scaffold_train_indices % 1000000)).tolist()
    scaffold_valid_indices = np.hstack((scaffold_valid_indices // 1000000, scaffold_valid_indices % 1000000)).tolist()
    scaffold_test_indices = np.hstack((scaffold_test_indices // 1000000, scaffold_test_indices % 1000000)).tolist()
    # Collecting random graphs
    if args.random_train:
        random_train_mols = []
        mols_count = 0
        for count, indices in enumerate(tqdm(random_train_indices, desc="Collecting random train mols", ncols=100, leave=False), 1):
            random_train_mols.append(supplier_ls[indices[0]][indices[1]])
            if count % 600000 == 0:
                with Chem.SDWriter(osp.join(args.out_dir, f"random/train_{mols_count}.sdf")) as w:
                    for mol in tqdm(random_train_mols, desc=f"Writing No.{mols_count} random train mols", ncols=100, leave=False):
                        w.write(mol)
                mols_count += 1
                del random_train_mols
                gc.collect()
                random_train_mols = []
        with Chem.SDWriter(osp.join(args.out_dir, f"random/train_{mols_count}.sdf")) as w:
            for mol in tqdm(random_train_mols, desc=f"Writing No.{mols_count} random train mols", ncols=100, leave=False):
                w.write(mol)
        del random_train_mols
        gc.collect()
    if args.random_valid:
        random_valid_mols = []
        for indices in tqdm(random_valid_indices, desc="Collecting random valid mols", ncols=100, leave=False):
            random_valid_mols.append(supplier_ls[indices[0]][indices[1]])
        with Chem.SDWriter(osp.join(args.out_dir, "random/valid.sdf")) as w:
            for mol in tqdm(random_valid_mols, desc="Writing random valid mols", ncols=100, leave=False):
                w.write(mol)
        del random_valid_mols
        gc.collect()
    if args.random_test:
        random_test_mols = []
        for indices in tqdm(random_test_indices, desc="Collecting random test mols", ncols=100, leave=False):
            random_test_mols.append(supplier_ls[indices[0]][indices[1]])
        with Chem.SDWriter(osp.join(args.out_dir, "random/test.sdf")) as w:
            for mol in tqdm(random_test_mols, desc="Writing random test mols", ncols=100, leave=False):
                w.write(mol)
        del random_test_mols
        gc.collect()
    # Collecting scaffold graphs
    if args.scaffold_train:
        scaffold_train_mols = []
        mols_count = 0
        for count, indices in enumerate(tqdm(scaffold_train_indices, desc="Collecting scaffold train mols", ncols=100, leave=False), 1):
            scaffold_train_mols.append(supplier_ls[indices[0]][indices[1]])
            if count % 600000 == 0:
                with Chem.SDWriter(osp.join(args.out_dir, f"scaffold/train_{mols_count}.sdf")) as w:
                    for mol in tqdm(scaffold_train_mols, desc=f"Writing No.{mols_count} scaffold train mols", ncols=100, leave=False):
                        w.write(mol)
                mols_count += 1
                del scaffold_train_mols
                gc.collect()
                scaffold_train_mols = []
        with Chem.SDWriter(osp.join(args.out_dir, f"scaffold/train_{mols_count}.sdf")) as w:
            for mol in tqdm(scaffold_train_mols, desc=f"Writing No.{mols_count} scaffold train mols", ncols=100, leave=False):
                w.write(mol)
        del scaffold_train_mols
        gc.collect()
    if args.scaffold_valid:
        scaffold_valid_mols = []
        for indices in tqdm(scaffold_valid_indices, desc="Collecting scaffold valid mols", ncols=100, leave=False):
            scaffold_valid_mols.append(supplier_ls[indices[0]][indices[1]])
        with Chem.SDWriter(osp.join(args.out_dir, "scaffold/valid.sdf")) as w:
            for mol in tqdm(scaffold_valid_mols, desc="Writing scaffold valid mols", ncols=100, leave=False):
                w.write(mol)
        del scaffold_valid_mols
        gc.collect()
    if args.scaffold_test:
        scaffold_test_mols = []
        for indices in tqdm(scaffold_test_indices, desc="Collecting scaffold test mols", ncols=100, leave=False):
            scaffold_test_mols.append(supplier_ls[indices[0]][indices[1]])
        with Chem.SDWriter(osp.join(args.out_dir, "scaffold/test.sdf")) as w:
            for mol in tqdm(scaffold_test_mols, desc="Writing scaffold test mols", ncols=100, leave=False):
                w.write(mol)
        del scaffold_test_mols
        gc.collect()
    print("Collecting graphs done.")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--out_dir", type=str, default="../data/split")
    parser.add_argument("--random_train", type=lambda x: True if x == "True" else False, default=False)
    parser.add_argument("--random_valid", type=lambda x: True if x == "True" else False, default=False)
    parser.add_argument("--random_test", type=lambda x: True if x == "True" else False, default=False)
    parser.add_argument("--scaffold_train", type=lambda x: True if x == "True" else False, default=True)
    parser.add_argument("--scaffold_valid", type=lambda x: True if x == "True" else False, default=False)
    parser.add_argument("--scaffold_test", type=lambda x: True if x == "True" else False, default=False)
    args = parser.parse_args()
    print(args)
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(osp.join(args.out_dir, "random"), exist_ok=True)
    os.makedirs(osp.join(args.out_dir, "scaffold"), exist_ok=True)
    process(args)
