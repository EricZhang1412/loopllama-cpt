## Before Running:
- [ ] Please check that there is a link to the model path (`models/{YOUR_MODEL_NAME}/{FILES}`)
- [ ] Check if you have the dataset folder: `data` that is organized like this:
```shell
.
├── data
│   ├── dataset_info.json
│   ├── slimpajama_jsonl
└───└── slimpajama_train_6B
```
whereas the `dataset_info.json` file is the registry entry point of the llamafactory project:
```json
{
  "slimpajama_train_6B": {
    "file_name": "slimpajama_train_6B",        // 这里填“目录名”，支持目录
    "columns": { "prompt": "text" }            // 预训练(PT)只需要把 text 列映射到 prompt
  }
}
```
- [ ] Install tensorboardX into the llamafactory docker container

## Model Path:
Please refer to [This Huggingface Repo](https://huggingface.co/ericzhang0328/loopllama-1B/tree/main) to get the current implementation of the looped Llama3.2-1B with KVMerge Technique.

## Script:

1. `deepspeed_cpt.sh`
```shell
deepspeed --num_gpus=8 --module llamafactory.launcher \
  --stage pt \
  --model_name_or_path models/llama3_2-1B \
  --dataset_dir data \
  --dataset slimpajama_train_6B \
  --output_dir output/llama3.2-1b-cpt-ds \
  --finetuning_type full \
  --do_train \
  --max_steps 30720 \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1.0e-5 \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.01 \
  --weight_decay 0.1 \
  --packing true \
  --cutoff_len 1024 \
  --max_samples 50000 \
  --bf16 true \
  --gradient_checkpointing true \
  --logging_steps 1 \
  --save_steps 1000 \
  --report_to tensorboard
```
2. `deepspeed_cpt_loopllama.sh`
