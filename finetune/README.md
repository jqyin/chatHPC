# Distributed Fine-tuning
Our development is based on the chat application in DeepSpeedExamples(`#32083e5`), and we add following additional features:

- support for [FORGE](https://github.com/at-aaims/forge), which are LLMs pre-trained on scientific publications
- support for ZeRO++, which enables communication compression
- custum dataset and launch support on Frontier 

# FORGE support 
The multi-node FORGE fine-tuning configurations are provided for [step 1](./step1_supervised_finetuning/training_scripts/forge/multi_node) and [step 3](./step3_rlhf_finetuning/training_scripts/forge/multi_node)

# ZeRO++ support
Add ZeRO++ and Flops profiler configuration in the main script, 
```bash
    ds_config['zero_optimization'] = {
        "stage": 3,
        "offload_param": {
            "device": "none"
        },
        "offload_optimizer": {
            "device": "none"
        },
        "stage3_param_persistence_threshold": 1.000000e+04,
        "stage3_max_live_parameters": 3.000000e+07,
        "stage3_prefetch_bucket_size": 3.000000e+07,
        "memory_efficient_linear": False,
        "allgather_partitions": True,
        "allgather_bucket_size": 500000000,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 500000000,
        "contiguous_gradients": True,
        "zero_quantized_weights": True,
        "zero_hpz_partition_size": 8,
        "zero_quantized_gradients": True 
      }
```

## Frontier
A [job script](./job-zero+.sb) is provided for Frontier, which runs step 1-3 as follows,
```bash
#step1
#python3 train.py --actor-model forge-m2-final --step 1 --deployment-type multi_node
#step2
#python3 train.py --reward-model forge-s4  --step 2 --deployment-type single_node
#step3
python3 train.py --actor-model forge-m2-final --reward-model forge-s4  --step 3 --deployment-type multi_node
```

The launch of deepspeed with slurm on Frontier,
```bash
deepspeed --launcher slurm --launcher_args "-n $((SLURM_JOB_NUM_NODES*8))  --ntasks-per-node=8 -c7 --gpus-per-node=8 --gpu-bind=closest" --force_multi
```
