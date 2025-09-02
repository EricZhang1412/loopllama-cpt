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
2. `deepspeed_cpt_loopllama.sh`
