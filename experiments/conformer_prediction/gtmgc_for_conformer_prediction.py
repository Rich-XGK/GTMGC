"""
    This script is used to train a Conformer model for conformer prediction.
"""

import os
import warnings
from argparse import Namespace
from datasets import load_dataset, DatasetDict
from dataclasses import dataclass

from transformers import EvalPrediction
from transformers import TrainingArguments, Trainer
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from models import GTMGCConfig, GTMGCForConformerPrediction, GTMGCCollator

warnings.simplefilter("ignore", UserWarning)


@dataclass
class ScriptArguments:
    tokenized_dataset_path: str


def compute_metrics(eval_pred: EvalPrediction):
    # EvalPrediction(predictions=all_preds, label_ids=all_labels) from transformers Trainer with args.include_inputs_for_metrics=False.
    preds, _ = eval_pred
    mae, mse, rmsd, *_ = preds
    mae, mse, rmsd = mae.mean().item(), mse.mean().item(), rmsd.mean().item()

    # print(f"MAE: {mae:.3f}, RMSE: {rmse:.3f}")
    return {"mae": mae, "mse": mse, "rmsd": rmsd}


def main():
    parse = HfArgumentParser(dataclass_types=[ScriptArguments, TrainingArguments])
    script_args, training_args = parse.parse_args_into_dataclasses()
    print(script_args)
    print(training_args)
    dataset = DatasetDict.load_from_disk(script_args.tokenized_dataset_path)
    train_set, eval_set, test_set = (
        dataset["train"],
        dataset["validation"],
        dataset["test"],
    )
    collate_func = GTMGCCollator(
        # max_nodes=200,
        # max_edges=300,
    )

    config = GTMGCConfig(
        # encoder config
        n_encode_layers=6,
        encoder_use_A_in_attn=True,
        encoder_use_D_in_attn=False,
        # decoder config
        n_decode_layers=6,
        decoder_use_A_in_attn=True,
        decoder_use_D_in_attn=True,
        embed_style="atom_tokenized_ids",
        # model config
        atom_vocab_size=513,  # 512 + 1 for padding (shifted all input ids by 1)
        # atom_vocab_size=119,  # 118 + 1 for padding (shifted all input ids by 1). For ablation: node type as IDs.
        d_embed=256,
        pre_ln=False,  # layer norm before residual, else after residual, not Pre-LN and not Post-LN.
        d_q=256,
        d_k=256,
        d_v=256,
        d_model=256,
        n_head=8,
        qkv_bias=True,
        attn_drop=0.00,
        norm_drop=0.00,
        ffn_drop=0.00,
        d_ffn=1024,
    )
    model = GTMGCForConformerPrediction(config)
    # model = GTMGCForConformerPrediction.from_pretrained('./checkpoints/Conformer_Prediction/GTMGC_Molecule3D_Random')
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
