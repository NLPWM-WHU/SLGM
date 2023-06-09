
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python run.py \
  --do_train True --do_predict --predict_with_generate \
  --overwrite_output_dir \
  --model_name_or_path=${model_name_or_path} \
  --dataset=${dataset} \
  --seed=${seed} \
  --output_dir=${output_dir} \
  --data_format=${data_format} \
  --per_device_train_batch_size=${train_batch_size} \
  --per_device_eval_batch_size=${eval_batch_size} \
  --learning_rate=${learning_rate} \
  --num_train_epochs=${epochs} \
  --save_strategy=${save_strategy} \
  --save_steps=${save_steps} \
  --lr_scheduler_type=linear \
  --use_marker=${use_marker} \
  --constraint_decoding ${constraint_decoding} \
  --alpha ${alpha} \
  --use_fast_tokenizer \
  --evaluation_strategy=${evaluation_strategy} \
  --eval_steps=${eval_steps} \
  --load_best_model_at_end=${load_best_model_at_end} \
  --metric_for_best_model eval_f1_score \
  --save_total_limit 1 \
  --shot_ratio_index=${shot_ratio_index} \
  --marker_type=${marker_type} \
  --warmup_ratio=${warmup_ratio}