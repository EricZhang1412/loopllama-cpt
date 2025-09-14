# deepspeed --num_gpus=8 --module llamafactory.launcher \
#   --stage pt \
#   --model_name_or_path models/llama3_2-1B \
#   --dataset_dir data \
#   --dataset slimpajama_train_6B \
#   --output_dir output/llama3.2-1b-cpt-ds \
#   --finetuning_type full \
#   --do_train \
#   --max_steps 30720 \
#   --per_device_train_batch_size 16 \
#   --gradient_accumulation_steps 4 \
#   --learning_rate 1.0e-5 \
#   --lr_scheduler_type cosine \
#   --warmup_ratio 0.01 \
#   --weight_decay 0.1 \
#   --packing true \
#   --cutoff_len 1024 \
#   --max_samples 50000 \
#   --bf16 true \
#   --gradient_checkpointing true \
#   --logging_steps 1 \
#   --save_steps 1000 \
#   --report_to tensorboard


#### from scratch ####
deepspeed --num_gpus=8 --module llamafactory.launcher \
  --stage pt \
  --model_name_or_path models/llama3_2-1B-scratch \
  --dataset_dir data \
  --dataset slimpajama_train_6B \
  --output_dir output/llama3.2-1b-from-scratch \
  --finetuning_type full \
  --do_train \
  --max_steps 150000 \
  --per_device_train_batch_size 24 \
  --gradient_accumulation_steps 4 \
  --learning_rate 3.0e-4 \
  --lr_scheduler_type cosine \
  --warmup_steps 2000 \
  --weight_decay 0.1 \
  --packing false \
  --bf16 true \
  --gradient_checkpointing true \
  --logging_steps 10 \
  --save_steps 5000 \
  --eval_steps 10000 \
  --save_total_limit 5 \
  --cutoff_len 1024 \
  --max_length 1024 \
  --preprocessing_num_workers 64 \
  --dataloader_num_workers 64 \
  --report_to tensorboard \
  --overwrite_output_dir true