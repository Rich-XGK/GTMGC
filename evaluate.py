import os
import torch
import warnings
import argparse
import numpy as np
import pandas as pd
import os.path as osp
import torch.nn.functional as F

from tqdm import tqdm
from rdkit import Chem
from copy import deepcopy
from rdkit import RDLogger
from typing import Literal
from rdkit.Chem import AllChem
from pprint import pprint, pformat
from rdkit.Chem import rdchem, rdMolAlign
from rdkit.Chem.rdmolfiles import SDMolSupplier

from molecule3d.utils import mol_to_graph_dict
from models.gtmgc import GTMGCCollator, GTMGCForConformerPrediction
from models.gnn import GNNCollator, GNNForConformerPrediction
from models.gps import GPSCollator, GPSForConformerPrediction
from models.mole_bert_tokenizer import MoleBERTTokenizerCollator, MoleBERTTokenizer

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def get_molecule3d_supplier(root_dir: str, mode: Literal["random", "scaffold"] = "random", split: Literal["valid", "test"] = "test"):
    sdf_files = osp.join(root_dir, mode, f"{split}.sdf")
    sdf_files = os.path.expanduser(sdf_files)
    if not osp.exists(sdf_files):
        raise FileNotFoundError(f"{sdf_files} does not exist.")
    supplier = Chem.SDMolSupplier(sdf_files, removeHs=False, sanitize=True)
    return supplier


def get_qm9_supplier(root_dir: str, split: Literal["valid", "test"] = "test"):
    sdf_files = osp.join(root_dir, f"gdb9.sdf")
    sdf_files = os.path.expanduser(sdf_files)
    if not osp.exists(sdf_files):
        raise FileNotFoundError(f"{sdf_files} does not exist.")
    supplier = Chem.SDMolSupplier(sdf_files, removeHs=False, sanitize=True)
    indices_df = pd.read_csv(osp.join(root_dir, f"{split}_indices.csv"))
    indices = indices_df["index"].tolist()
    supplier = [supplier[i] for i in indices]
    return supplier


def get_metrics(mol: rdchem.Mol, mol_h: rdchem.Mol = None, R: np.ndarray = None, R_h: np.ndarray = None, removeHs: bool = False):
    if mol_h is None and R_h is None:
        raise ValueError("mol_h and R_h cannot both be None.")
    if mol_h is not None and R_h is not None:
        raise ValueError("mol_h and R_h cannot both be not None.")
    if mol is None and R is not None:
        pass
    if mol_h is None and R_h is not None:
        mol_h = deepcopy(mol)
        # mol_h = Chem.AddHs(mol)
        R_h = R_h.tolist()
        conf_h = rdchem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf_h.SetAtomPosition(i, R_h[i])
        mol_h.RemoveConformer(0)
        mol_h.AddConformer(conf_h)
    R, R_h = mol.GetConformer().GetPositions(), mol_h.GetConformer().GetPositions()
    R, R_h = torch.from_numpy(R), torch.from_numpy(R_h)
    D, D_h = torch.cdist(R, R), torch.cdist(R_h, R_h)
    mae = F.l1_loss(D, D_h, reduction="sum").item()
    mse = F.mse_loss(D, D_h, reduction="sum").item()
    num_dist = D.numel()
    if removeHs:
        try:
            mol, mol_h = Chem.RemoveHs(mol), Chem.RemoveHs(mol_h)
        except Exception as e:
            pass
    rmsd = rdMolAlign.GetBestRMS(mol, mol_h)
    return {
        "mae": mae,
        "mse": mse,
        "rmsd": rmsd,
        "num_dist": num_dist,
    }


def evaluate_gtmgc(
    GTMGC: GTMGCForConformerPrediction,
    tokenizer: MoleBERTTokenizer,
    GTMGC_collator: GTMGCCollator,
    tokenizer_collator: MoleBERTTokenizerCollator,
    supplier: SDMolSupplier,
    device: torch.device = "cpu",
    batch_size: int = 300,
    removeHs: bool = False,
):
    num_mol = len(supplier)
    num_batch = num_mol // batch_size + 1
    GTMGC.eval()
    total_mae, total_mse, total_dist, total_rmsd = 0.0, 0.0, 0.0, 0.0
    inf_pro_bar = tqdm(total=num_batch, desc="Inference on GTMGC", ncols=100, leave=False)
    eval_pro_bar = tqdm(total=num_mol, desc="Evaluation", ncols=100, leave=False)
    num_mol = 0
    for i in range(num_batch):
        start, end = (i * batch_size, (i + 1) * batch_size) if i < num_batch - 1 else (i * batch_size, num_mol)
        mol_ls = [supplier[j] for j in range(start, end) if supplier[j] is not None]
        if not mol_ls:
            continue
        num_mol += len(mol_ls)
        mol_dict_ls = []
        for mol in mol_ls:
            mol_dict = mol_to_graph_dict(mol)
            batch = tokenizer_collator([mol_dict])
            batch = {k: v.to(device) for k, v in batch.items()}
            out = tokenizer(**batch)
            mol_dict["input_ids"] = out["quantized_indices"].tolist()
            mol_dict_ls.append(mol_dict)
        batch = GTMGC_collator(mol_dict_ls)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = GTMGC(**batch)
        padding_mask = batch["node_mask"]
        R_h = out["conformer_hat"]
        batch_rmsd = 0.0
        # compute coord rmsd
        for j, mol in enumerate(mol_ls):
            R_h_i = R_h[j]
            mask = padding_mask[j]
            R_h_i = R_h_i[mask == 1].detach().cpu().numpy()
            metrics = get_metrics(mol=mol, R_h=R_h_i, removeHs=removeHs)
            mae, mse, rmsd, num_dist = metrics["mae"], metrics["mse"], metrics["rmsd"], metrics["num_dist"]
            total_mae += mae
            total_mse += mse
            total_dist += num_dist
            total_rmsd += rmsd
            batch_rmsd += rmsd
            eval_pro_bar.set_postfix({"mae": f"{mae/num_dist:.3f}", "rmse": f"{np.sqrt(mse/num_dist):.3f}", "rmsd": f"{rmsd:.3f}"})
            eval_pro_bar.update()
        inf_pro_bar.set_postfix({"batch": i, "rmsd": f"{batch_rmsd/len(mol_ls):.3f}"})
        inf_pro_bar.update()

    mae = total_mae / total_dist
    rmse = np.sqrt(total_mse / total_dist)
    rmsd = total_rmsd / num_mol
    return {"mae": mae, "rmse": rmse, "rmsd": rmsd}


def evaluate_gnn(
    GNN: GNNForConformerPrediction,
    GNN_collator: GNNCollator,
    supplier: SDMolSupplier,
    device: torch.device = "cpu",
    batch_size: int = 300,
    removeHs: bool = False,
):
    num_mol = len(supplier)
    num_batch = num_mol // batch_size + 1
    GNN.eval()
    total_mae, total_mse, total_dist, total_rmsd = 0.0, 0.0, 0.0, 0.0
    inf_pro_bar = tqdm(total=num_batch, desc="Inference on GNN", ncols=100, leave=False)
    eval_pro_bar = tqdm(total=num_mol, desc="Evaluation", ncols=100, leave=False)
    num_mol = 0
    for i in range(num_batch):
        start, end = (i * batch_size, (i + 1) * batch_size) if i < num_batch - 1 else (i * batch_size, num_mol)
        mol_ls = [supplier[j] for j in range(start, end) if supplier[j] is not None]
        if not mol_ls:
            continue
        num_mol += len(mol_ls)
        batch = [mol_to_graph_dict(mol) for mol in mol_ls]
        batch = GNN_collator(batch)
        graph_index = batch["batch"]
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = GNN(**batch)
        R_h = out["conformer_hat"]
        batch_rmsd = 0.0
        # compute coord rmsd
        for j, mol in enumerate(mol_ls):
            R_h_i = R_h[graph_index == j].detach().cpu().numpy()
            metrics = get_metrics(mol=mol, R_h=R_h_i, removeHs=removeHs)
            mae, mse, rmsd, num_dist = metrics["mae"], metrics["mse"], metrics["rmsd"], metrics["num_dist"]
            total_mae += mae
            total_mse += mse
            total_dist += num_dist
            total_rmsd += rmsd
            batch_rmsd += rmsd
            eval_pro_bar.set_postfix({"mae": f"{mae/num_dist:.3f}", "rmse": f"{np.sqrt(mse/num_dist):.3f}", "rmsd": f"{rmsd:.3f}"})
            eval_pro_bar.update()
        inf_pro_bar.set_postfix({"batch": i, "rmsd": f"{batch_rmsd/len(mol_ls):.3f}"})
        inf_pro_bar.update()

    mae = total_mae / total_dist
    rmse = np.sqrt(total_mse / total_dist)
    rmsd = total_rmsd / num_mol
    return {"mae": mae, "rmse": rmse, "rmsd": rmsd}


def evaluate_rdkit(supplier: SDMolSupplier, method: Literal["DG", "ETKDG"] = "ETKDG", removeHs: bool = False):
    num_mol = len(supplier)
    total_mae, total_mse, total_dist, total_rmsd = 0.0, 0.0, 0.0, 0.0
    pro_bar = tqdm(supplier, desc="Evaluation", ncols=100, leave=False)
    for mol in supplier:
        if mol is None:
            continue
        mol_h = deepcopy(mol)
        # mol_h = Chem.AddHs(mol)   # ERROR: num atoms not match
        if method == "ETKDG":
            flag = AllChem.EmbedMolecule(mol_h, useRandomCoords=False)
        elif method == "DG":
            flag = AllChem.EmbedMolecule(mol_h, useRandomCoords=False, useExpTorsionAnglePrefs=False, useBasicKnowledge=False)
        # compute metrics
        if flag == -1:  # fail to embed
            num_mol -= 1
            pro_bar.update()
            continue
        metrics = get_metrics(mol=mol, mol_h=mol_h, removeHs=removeHs)
        mae, mse, rmsd, num_dist = metrics["mae"], metrics["mse"], metrics["rmsd"], metrics["num_dist"]
        total_mae += mae
        total_mse += mse
        total_dist += num_dist
        total_rmsd += rmsd
        pro_bar.update()
        pro_bar.set_postfix({"mae": f"{mae/num_dist:.3f}", "rmse": f"{np.sqrt(mse/num_dist):.3f}", "rmsd": f"{rmsd:.3f}"})

    mae = total_mae / total_dist
    rmse = np.sqrt(total_mse / total_dist)
    rmsd = total_rmsd / num_mol
    return {"mae": mae, "rmse": rmse, "rmsd": rmsd}


if __name__ == "__main__":
    parse = argparse.ArgumentParser("Conformation Prediction Evaluation")
    parse.add_argument("--data_dir", type=str, default="~/DataSets/", help="The root directory of the dataset.")
    parse.add_argument("--dataset", type=str, default="Molecule3D", choices=["Molecule3D", "Qm9"], help="The dataset to be evaluated.")
    parse.add_argument("--mode", type=str, default="random", choices=["random", "scaffold"], help="The mode of Molecule3D.")
    parse.add_argument("--split", type=str, default="test", choices=["valid", "test"], help="The split of Molecule3D.")
    parse.add_argument("--seed", type=int, default=42, help="The random seed.")
    parse.add_argument("--log_file", type=str, default="./evaluate.txt", help="The log file to save the evaluation results.")
    parse.add_argument("--method", type=str, default="GTMGC", choices=["Rdkit-DG", "Rdkit-ETKDG", "GTMGC", "GNN", "GPS"], help="The method to be evaluated.")
    parse.add_argument("--removeHs", type=lambda x: str(x).lower() == "true", default="true", help="Whether to remove Hs.")
    parse.add_argument(
        "--GTMGC_checkpoint", type=str, default="./checkpoints/CP/GTMGC_Molecule3D_Random", help="The checkpoint of GTMGC."
    )
    parse.add_argument(
        "--GNN_checkpoint", type=str, default="./checkpoints/CP/GINE_Molecule3D_Random", help="The checkpoint of GNN."
    )
    parse.add_argument(
        "--GPS_checkpoint", type=str, default="./checkpoints/CP/GPS_Molecule3D_Random", help="The checkpoint of GPS."
    )
    parse.add_argument("--MoleBERT_Tokenizer_checkpoint", type=str, default="RichXuOvO/MoleBERT-Tokenizer", help="The checkpoint of MoleBERT_Tokenizer.")
    parse.add_argument("--device", type=str, default="cuda:1", help="The device to run evaluation on.")
    parse.add_argument("--batch_size", type=int, default=100, help="The batch size to run evaluation.")

    args = vars(parse.parse_args())
    pprint(args)

    # make log file
    os.makedirs(osp.dirname(args["log_file"]), exist_ok=True)
    # set random seed
    torch.manual_seed(args["seed"])
    np.random.seed(args["seed"])

    # get supplier
    supplier = None
    if args["dataset"] == "Molecule3D":
        root_dir = osp.join(args["data_dir"], "Molecule3D")
        supplier = get_molecule3d_supplier(root_dir=root_dir, mode=args["mode"], split=args["split"])
    elif args["dataset"] == "Qm9":
        root_dir = osp.join(args["data_dir"], "Qm9")
        supplier = get_qm9_supplier(root_dir=root_dir, split=args["split"])

    # evaluate
    if args["method"] == "Rdkit-DG":
        metrics = evaluate_rdkit(supplier=supplier, method="DG", removeHs=args["removeHs"])
    elif args["method"] == "Rdkit-ETKDG":
        metrics = evaluate_rdkit(supplier=supplier, method="ETKDG", removeHs=args["removeHs"])
    elif args["method"] == "GTMGC":
        device = torch.device(args["device"])
        tokenizer_collator = MoleBERTTokenizerCollator()
        GTMGC_collator = GTMGCCollator()
        tokenizer = MoleBERTTokenizer.from_pretrained(args["MoleBERT_Tokenizer_checkpoint"]).to(device)
        print(tokenizer.config)
        GTMGC = GTMGCForConformerPrediction.from_pretrained(args["GTMGC_checkpoint"]).to(device)
        print(GTMGC.config)
        metrics = evaluate_gtmgc(
            GTMGC=GTMGC,
            tokenizer=tokenizer,
            GTMGC_collator=GTMGC_collator,
            tokenizer_collator=tokenizer_collator,
            supplier=supplier,
            device=device,
            batch_size=args["batch_size"],
            removeHs=args["removeHs"],
        )
    elif args["method"] == "GNN":
        device = torch.device(args["device"])
        GPS = GNNForConformerPrediction.from_pretrained(args["GNN_checkpoint"]).to(device)
        GPS_collator = GNNCollator()
        print(GPS.config)
        metrics = evaluate_gnn(
            GNN=GPS,
            GNN_collator=GPS_collator,
            supplier=supplier,
            device=device,
            batch_size=args["batch_size"],
            removeHs=args["removeHs"],
        )
    elif args["method"] == "GPS":
        device = torch.device(args["device"])
        GPS = GPSForConformerPrediction.from_pretrained(args["GPS_checkpoint"]).to(device)
        GPS_collator = GPSCollator("random_walk", 10)
        print(GPS.config)
        metrics = evaluate_gnn(
            GNN=GPS,
            GNN_collator=GPS_collator,
            supplier=supplier,
            device=device,
            batch_size=args["batch_size"],
            removeHs=args["removeHs"],
        )

    info = f"mae: {metrics['mae']:.4f}, rmse: {metrics['rmse']:.4f}, rmsd: {metrics['rmsd']:.4f}\n"
    info += pformat(args)
    print(info)
    with open(args["log_file"], "a") as f:
        f.write(info)
        f.write("\n")
