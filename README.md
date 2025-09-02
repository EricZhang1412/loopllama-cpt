## Before Running:
- [ ] Please check that there is a link to the model path (`models/{YOUR_MODEL_NAME}/{FILES}`)
- [ ] Check if you have the dataset folder that is organized like this:
- [ ] Install tensorboardX into the llamafactory docker container

## Model Path:
Please refer to [This Huggingface Repo](https://huggingface.co/ericzhang0328/loopllama-1B/tree/main) to get the current implementation of the looped Llama3.2-1B with KVMerge Technique.

## Script:

1. `deepspeed_cpt.sh`
2. `deepspeed_cpt_loopllama.sh`
