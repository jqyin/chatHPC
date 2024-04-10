#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
OUTPUT=$1
ZERO_STAGE=$2
if [ "$OUTPUT" == "" ]; then
    OUTPUT=./output
fi
if [ "$ZERO_STAGE" == "" ]; then
    ZERO_STAGE=2
fi
mkdir -p $OUTPUT

#deepspeed --launcher slurm --launcher_args "-n 16 --ntasks-per-node=8 -c7 --gpus-per-node=8 --gpu-bind=closest" --force_multi \
#    main.py --local_rank 0 \
deepspeed  \
    main.py \
   --sft_only_data_path local/jsonfile \
   --data_split 2,4,4 \
   --model_name_or_path forge-s4 \
   --per_device_train_batch_size 2 \
   --per_device_eval_batch_size 2 \
   --max_seq_len 1024 \
   --learning_rate 9.65e-6 \
   --weight_decay 0. \
   --num_train_epochs 10 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type cosine \
   --num_warmup_steps 0 \
   --zero_stage 0 \
   --gradient_checkpointing \
   --seed 1234 \
   --deepspeed \
   --enable_tensorboard \
   --tensorboard_path $OUTPUT \
   --output_dir $OUTPUT \
   &> $OUTPUT/training.log
   #--zero_stage $ZERO_STAGE \
