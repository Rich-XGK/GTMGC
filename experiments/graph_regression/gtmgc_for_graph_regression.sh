#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES='0' \
    python \
    -m experiments.graph_regression.gtmgc_for_graph_regression \
    --dataset_name=Molecule3D_Random \
    --evaluation_only_checkpoint=./checkpoints/GR/Molecule3D_Gap_Large \
    --output_dir=./results/graph_regression/GR_GTMGC_Molecule3D_Random_Gap \
    --run_name=GR_GTMGC_Molecule3D_Random_Gap \
    --resume_from_checkpoint=False \
    --overwrite_output_dir=True \
    --do_train=True \
    --do_eval=True \
    --do_predict=False \
    --evaluation_strategy=epoch \
    --save_strategy=epoch \
    --save_steps=1000 \
    --save_total_limit=10 \
    --logging_strategy=epoch \
    --logging_steps=1000 \
    --prediction_loss_only=False \
    --per_device_train_batch_size=100 \
    --per_device_eval_batch_size=100 \
    --gradient_accumulation_steps=1 \
    --eval_delay=0 \
    --learning_rate=5e-5 \
    --optim=adamw_torch \
    --weight_decay=0.0 \
    --adam_beta1=0.9 \
    --adam_beta2=0.99 \
    --adam_epsilon=1e-8 \
    --max_grad_norm=5.0 \
    --num_train_epochs=60 \
    --max_steps=-1 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.3 \
    --warmup_steps=0 \
    --log_level=info \
    --seed=42 \
    --data_seed=42 \
    --use_ipex=False \
    --bf16=False \
    --fp16=True \
    --fp16_opt_level=O1 \
    --half_precision_backend=auto \
    --bf16_full_eval=False \
    --fp16_full_eval=True \
    --tf32=False \
    --dataloader_drop_last \
    --dataloader_num_workers=8 \
    --disable_tqdm=False \
    --remove_unused_columns=False \
    --label_names=labels \
    --load_best_model_at_end=True \
    --metric_for_best_model=mae \
    --greater_is_better=False \
    --push_to_hub=False \
    --include_inputs_for_metrics=False \
    --torch_compile=False \
    --report_to=wandb
