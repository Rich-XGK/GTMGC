"""
    This file contains the code for tokenizing the molecules.
"""
import os
import argparse
import os.path as osp
from typing import Dict, Any
from molecule3d import HFMolecule3DRandomSplit, HFMolecule3DScaffoldSplit, HFQm9
from models.mole_bert_tokenizer import MoleBERTTokenizerCollator, MoleBERTTokenizer


def tokenize(mol_dict: Dict[str, Any], tokenizer: MoleBERTTokenizer, collator: MoleBERTTokenizerCollator):
    """
    Tokenize the molecules.
    """
    batch = collator([mol_dict])
    out = tokenizer(**batch)
    input_ids = out["quantized_indices"].tolist()
    mol_dict["input_ids"] = input_ids
    return mol_dict


def main(args: Dict[str, Any]):
    if args["dataset_name"] == "Molecule3D":
        if args["mode"] == "random":
            dataset_builder = HFMolecule3DRandomSplit(use_auth_token=True)
        elif args["mode"] == "scaffold":
            dataset_builder = HFMolecule3DScaffoldSplit(use_auth_token=True)
        save_dir = osp.join(args["save_dir"], args["mode"])
    elif args["dataset_name"] == "Qm9":
        dataset_builder = HFQm9(use_auth_token=True)
        save_dir = args["save_dir"]
    else:
        raise ValueError("The dataset name is not supported.")
    if not osp.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    tokenizer_collator = MoleBERTTokenizerCollator()
    tokenizer = MoleBERTTokenizer.from_pretrained(args["tokenizer_checkpoint"])
    print(tokenizer)
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    unused_columns = list(set(args["unused_columns"]) & set(dataset.column_names['train']))
    print(f"Unused columns: {unused_columns}")
    dataset = dataset.map(lambda x: tokenize(x, tokenizer, tokenizer_collator))
    dataset = dataset.remove_columns(unused_columns)
    dataset.save_to_disk(save_dir)


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Tokenize Molecule3D DataSet.")
    parse.add_argument("--save_dir", type=str, default="/home/Xgk/DataSets/Tokenized_HFMolecule3D", help="The directory to save the tokenized dataset.")
    parse.add_argument("--dataset_name", type=str, default="Molecule3D", choices=["Molecule3D", "Qm9"], help="The name of the dataset.")
    parse.add_argument("--mode", type=str, default="random", choices=["random", "scaffold"], help="The mode of Molecule3D.")
    parse.add_argument("--tokenizer_checkpoint", type=str, default="RichXuOvO/MoleBERT-Tokenizer", help="The checkpoint of MoleBERT_Tokenizer.")
    parse.add_argument(
        "--unused_columns",
        nargs="+",
        default=["node_chiral_type", "edge_type", "edge_dire_type", "num_nodes", "num_edges", "cid", "dipole x", "dipole y", "dipole z"],
        help="The columns to be removed.",
    )
    args = vars(parse.parse_args())
    main(args)
