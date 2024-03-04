"""
    This script is used to train a Conformer model for conformer prediction.
"""
import warnings
from argparse import Namespace
from datasets import load_dataset, DatasetDict
from molecule3d import HFMolecule3DRandomSplit, HFMolecule3DScaffoldSplit, HFQm9
from transformers import EvalPrediction
from transformers import TrainingArguments, Trainer
from transformers import HfArgumentParser

from models import GNNCollator, GNNConfig, GNNForConformerPrediction

warnings.simplefilter("ignore", UserWarning)
parse = HfArgumentParser(dataclass_types=[TrainingArguments])
args = parse.parse_args()
print(args)


def compute_metrics(eval_pred: EvalPrediction):
    # EvalPrediction(predictions=all_preds, label_ids=all_labels) from transformers Trainer with args.include_inputs_for_metrics=False.
    preds, _ = eval_pred
    mae, *_ = preds
    mae = mae.mean().item()

    # print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    return {"mae": mae}


def main(args: Namespace):
    dataset_builder = HFQm9(use_auth_token=True)
    # dataset_builder = HFMolecule3DRandomSplit(use_auth_token=True)
    # dataset_builder = HFMolecule3DScaffoldSplit(use_auth_token=True)
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    train_set, eval_set, test_set = dataset["train"], dataset["validation"], dataset["test"]
    collate_func = GNNCollator()

    config = GNNConfig(num_layers=8, d_embed=512, dropout=0.0, JK="last", gnn_type="gatv2", gat_num_heads=8)

    model = GNNForConformerPrediction(config)
    print(model.config)
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
