"""
    This script is used to train a MoleBERT tokenizer.
"""

import warnings
import os.path as osp
from argparse import Namespace
from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from transformers import HfArgumentParser
from models import MoleBERTTokenizerCollator, MoleBERTTokenizerConfig, MoleBERTTokenizerForGraphReconstruct
from molecule3d.molecule3d import HFMolecule3DRandomSplit


warnings.filterwarnings("ignore", category=UserWarning)
parse = HfArgumentParser(dataclass_types=[TrainingArguments])
args = parse.parse_args()


def compute_metrics(eval_pred):
    preds, _ = eval_pred
    acc_recon = preds[-1]
    acc_recon = acc_recon.mean().item()
    return {"acc_recon": acc_recon}


def main(args: Namespace):
    dataset_builder = HFMolecule3DRandomSplit(use_auth_token=True)
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    train_set, eval_set, test_set = dataset["train"], dataset["validation"], dataset["test"]
    collate_func = MoleBERTTokenizerCollator()
    config = MoleBERTTokenizerConfig(
        atom_vocab_size=1024,
    )
    model = MoleBERTTokenizerForGraphReconstruct(config)
    print(model.tokenizer)
    args = TrainingArguments(**args.__dict__)

    trainer = Trainer(
        model=model,
        args=args,
        data_collator=collate_func,
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.evaluate(eval_dataset=test_set)


if __name__ == "__main__":
    main(args)
