"""
    This script is used to train a Conformer model for graph regression.
"""

import warnings
from argparse import Namespace
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass
from typing import Literal, Union

from molecule3d import HFMolecule3DRandomSplit, HFQm9

from transformers import EvalPrediction
from transformers import TrainingArguments, Trainer
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint


from models import GTMGCConfig, GTMGCForGraphRegression, GTMGCCollator

warnings.simplefilter("ignore", UserWarning)


@dataclass
class ScriptArguments:
    dataset_name: Literal["Molecule3D_Random", "QM9"] = "Molecule3D_Random"
    evaluation_only_checkpoint: str = "./checkpoints/GR/Molecule3D_Gap_Big"


def compute_metrics(eval_pred: EvalPrediction):
    # EvalPrediction(predictions=all_preds, label_ids=all_labels) from transformers Trainer with args.include_inputs_for_metrics=False.
    preds, _ = eval_pred
    mae, mse, *_ = preds
    mae, mse = mae.mean().item(), mse.mean().item()
    return {"mae": mae, "mse": mse}


def main():
    parse = HfArgumentParser(dataclass_types=[ScriptArguments, TrainingArguments])
    script_args, training_args = parse.parse_args_into_dataclasses()
    if script_args.dataset_name == "Molecule3D_Random":
        dataset_builder = HFMolecule3DRandomSplit(use_auth_token=True)
    elif script_args.dataset_name == "QM9":
        dataset_builder = HFQm9(use_auth_token=True)
    else:
        raise ValueError("Invalid dataset name.")
    dataset_builder.download_and_prepare()
    dataset = dataset_builder.as_dataset()
    # Molecule3D: homolumogap, homo, lumo
    # QM9: alpha, gap, homo, lumo, cv, mu
    dataset = dataset.rename_column("homolumogap", "labels")
    print(dataset)
    train_set, eval_set, test_set = (
        dataset["train"],
        dataset["validation"],
        dataset["test"],
    )
    collate_func = GTMGCCollator()
    config = GTMGCConfig(
        # encoder config
        n_encode_layers=6,
        encoder_use_A_in_attn=True,
        encoder_use_D_in_attn=True,
        embed_style='ogb',
        # model config
        atom_vocab_size=513,  # 512 + 1 for padding (shifted all input ids by 1)
        d_embed=256,
        pre_ln=False,  # layer norm before residual, else after residual, not Pre-LN and not Post-LN.
        d_q=256,
        d_k=256,
        d_v=256,
        d_model=256,
        n_head=8,
        qkv_bias=True,
        attn_drop=0.0,
        norm_drop=0.0,
        ffn_drop=0.0,
        d_ffn=1024,
    )
    if training_args.do_train:
        model = GTMGCForGraphRegression(config)
    if not training_args.do_train and training_args.do_eval:
        model = GTMGCForGraphRegression.from_pretrained(
            script_args.evaluation_only_checkpoint
        )
    print(model.config)

    trainer = Trainer(
        model=model, 
        args=training_args,
        data_collator=collate_func,
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics,
    )

    # training
    if training_args.do_train:
        resume_from_checkpoint = training_args.resume_from_checkpoint
        try:
            resume_f = eval(resume_from_checkpoint)
            if not resume_f:
                train_result = trainer.train()
            else:
                last_checkpoint = get_last_checkpoint(training_args.output_dir)
                train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        except Exception:
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.log_metrics("train", train_result.metrics)

    # testing
    if training_args.do_eval:
        # TODO: if training_args.do_train is False, model used to test is not the best!
        test_metrics = trainer.evaluate(eval_dataset=test_set)
        trainer.log_metrics("test", test_metrics)


if __name__ == "__main__":
    main()
