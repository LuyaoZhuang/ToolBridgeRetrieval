ToolBridgeRetrieval: Aligning Vague Instructions with Retriever
Preferences via a Bridge Model
===========================

****
## 目录
* [Requirement](##Requirement)
* [Data](##Data)
* [Model](#Model)


## Requirement

Run the following command to install the dependencies:

```
pip install -r requirements.txt
```

## Data
Please download our dataset, VGToolBench using the following link: [Google Drive](https://drive.google.com/drive/folders/1xd7nQEodULkk-XHsNnf9PwnU9Vpsx-8I?usp=sharing).

The file structure is as follows:

```
├── /VGToolBench/
│  ├── /G1_cat/
│  │  ├── /hybrid/
│  │  └── /fuzzy/
│  ├── /G1_ins/
│  │  ├── /hybrid/
│  │  └── /fuzzy/
│  ├── /G1_tool/
│  │  ├── /hybrid/
│  │  └── /fuzzy/
│  ├── /G2_cat/
│  │  ├── /hybrid/
│  │  └── /fuzzy/
│  ├── /G2_ins/
│  │  ├── /hybrid/
│  │  └── /fuzzy/
│  └── /G3_cat/
│     ├── /hybrid/
│     └── /fuzzy/

```
**Data Description:**
- **`hybrid/`**: Contains data from the original ToolBench dataset
- **`fuzzy/`**: Contains data from our proposed VGToolBench dataset

# Model

## Train Bridge Model, Stage 1 SFT
### Code
```
#!/bin/bash
export CUDA_HOME=/usr/local/cuda
cd ./code/LLaMA-Factory

# Configuration
model_name=Llama-3.2-3B
dataset=G3_ins
exp_name=fuzzy_hybrid
lr=5e-5
output_dir=./output/${dataset}/bridge_model/${exp_name}_sft_${model_name}_lr_${lr}

# Setup output directory and logging
mkdir -p $output_dir
train_log=$output_dir/training.log
export WANDB_DISABLED=True

# Execute distributed training
CUDA_VISIBLE_DEVICES=1,3 python -m torch.distributed.run \
  --nproc_per_node 2\
  --nnodes 1 \
  --node_rank 0 \
  --master_addr "localhost" \
  --master_port 12105 \
  ./src/train.py \
  --model_name_or_path ./BridgeRetrieval/model/orig/${model_name} \
  --stage sft \
  --do_train true \
  --finetuning_type lora \
  --lora_target all \
  --output_dir $output_dir \
  --overwrite_cache False \
  --dataset "${dataset}_hybrid_sft" \
  --template llama3 \
  --cutoff_len 1024 \
  --per_device_train_batch_size 6 \
  --gradient_accumulation_steps 1 \
  --lr_scheduler_type cosine \
  --logging_steps 10 \
  --val_size 100 \
  --per_device_eval_batch_size 6 \
  --eval_strategy steps \
  --eval_steps 50000 \
  --save_steps 1000 \
  --learning_rate ${lr} \
  --num_train_epochs 3.0 \
  --plot_loss \
  --bf16 2>&1
```

## Train ToolPlanner, Stage 2 Reinforcement Learning
### Code
```
#!/bin/bash
# Configuration
model_name=Llama-3.2-3B
sft_step=6243
retriever_type=ToolRetriever
export CUDA_HOME=/usr/local/cuda
cd ./code/LLaMA-Factory

lr=2e-5
dataset=G3_ins
beta=0.2

# Setup output directory
output_dir=./output/${dataset}/bridge_model/fuzzy_hybrid_dpo_${model_name}_beta_${beta}_lr_${lr}/${retriever_type}_retriever_score/
mkdir -p $output_dir
train_log=$output_dir/training.log

echo "Starting DPO training with beta = ${beta}"

# Execute distributed DPO training
CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.run \
    --nproc_per_node 3 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr "localhost" \
    --master_port 12330 \
    ./src/train.py \
    --model_name_or_path ./model/${dataset}/bridge_model/fuzzy_hybrid_sft_${model_name}_lr_5e-5/checkpoint-${sft_step} \
    --stage dpo \
    --do_train true \
    --finetuning_type lora \
    --lora_target all \
    --pref_loss sigmoid \
    --dataset ${dataset}_${model_name}_${retriever_type}_dpo \
    --template llama3 \
    --cutoff_len 1024 \
    --output_dir $output_dir \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --logging_steps 10 \
    --plot_loss \
    --overwrite_output_dir true \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate ${lr} \
    --num_train_epochs 3.0 \
    --warmup_ratio 0.02 \
    --pref_beta ${beta} \
    --bf16 \
    --use_unsloth_gc True \
    --val_size 0.1 \
    --per_device_eval_batch_size 2 \
    --eval_strategy steps \
    --report_to wandb \
    --eval_steps 5000 2>&1 | tee ${train_log}
```
