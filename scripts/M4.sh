accelerate launch --config_file accelerrate_config.yaml --main_process_port 10097 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ../dataset/m4 \
  --model ATM \
  --data m4 \
  --seasonal_patterns Yearly \
  --features M \
  --factor 3 \
  --enc_in 1 \
  --itr 1 \
  --d_model 32 \
  --inter_dim 32 \
  --n_heads 8 \
  --n_querys 6 \
  --prob_bias 0.00 \
  --prob_bias_end 70 \
  --high_freq 1.5 \
  --threshold_ratio 0.991 \
  --aux_loss 1 \
  --aux_loss_factor 0.07 \
  --dropout 0.2 \
  --head_dropout 0.1 \
  --batch_size 36 \
  --learning_rate 0.0015 \
  --conv_layers 1 \
  --d_layers 2 \
  --num_experts 8 \
  --top_k 2 \
  --use_moe 1 \
  --apply_router_aux_loss 1 \
  --router_aux_loss_factor 0.02 \
  --train_epochs 15 \
  --patience 3 \
  --num_workers 4

for r in 0.1 0.2 0.3 0.4
do
accelerate launch --config_file accelerrate_config.yaml --main_process_port 10097 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ../dataset/m4 \
  --model ATM \
  --data m4 \
  --seasonal_patterns Quarterly \
  --features M \
  --factor 3 \
  --enc_in 1 \
  --itr 1 \
  --d_model 64 \
  --inter_dim 32 \
  --n_heads 8 \
  --n_querys 6 \
  --prob_bias 0.00 \
  --prob_bias_end 70 \
  --high_freq 3 \
  --threshold_ratio 0.991 \
  --aux_loss 1 \
  --aux_loss_factor $r \
  --dropout 0.05 \
  --head_dropout 0.1 \
  --batch_size 36 \
  --learning_rate 0.004 \
  --conv_layers 3 \
  --d_layers 1 \
  --num_experts 8 \
  --top_k 2 \
  --use_moe 1 \
  --apply_router_aux_loss 1 \
  --router_aux_loss_factor 0.02 \
  --train_epochs 20 \
  --patience 3 \
  --num_workers 4
done

accelerate launch --config_file accelerrate_config.yaml --main_process_port 10097 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ../dataset/m4 \
  --model ATM \
  --data m4 \
  --seasonal_patterns Monthly \
  --features M \
  --factor 3 \
  --enc_in 1 \
  --itr 1 \
  --d_model 32 \
  --inter_dim 32 \
  --n_heads 8 \
  --n_querys 6 \
  --prob_bias 0.00 \
  --prob_bias_end 70 \
  --high_freq 1.5 \
  --threshold_ratio 0.991 \
  --aux_loss 1 \
  --aux_loss_factor 0.2 \
  --dropout 0.05 \
  --head_dropout 0.1 \
  --batch_size 36 \
  --learning_rate 0.003 \
  --conv_layers 2 \
  --d_layers 1 \
  --num_experts 8 \
  --top_k 2 \
  --use_moe 1 \
  --apply_router_aux_loss 1 \
  --router_aux_loss_factor 0.02 \
  --train_epochs 8 \
  --patience 3 \
  --num_workers 4

for f in 0.5 0.6 0.7 0.8
do 
accelerate launch --config_file accelerrate_config.yaml --main_process_port 10097 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ../dataset/m4 \
  --model ATM \
  --data m4 \
  --seasonal_patterns Monthly \
  --features M \
  --factor 3 \
  --enc_in 1 \
  --itr 1 \
  --d_model 32 \
  --inter_dim 32 \
  --n_heads 8 \
  --n_querys 6 \
  --prob_bias 0.00 \
  --prob_bias_end 70 \
  --high_freq 1.5 \
  --threshold_ratio 0.991 \
  --aux_loss 1 \
  --aux_loss_factor $f \
  --dropout 0.05 \
  --head_dropout 0.1 \
  --batch_size 36 \
  --learning_rate 0.006 \
  --conv_layers 1 \
  --d_layers 1 \
  --num_experts 8 \
  --top_k 2 \
  --use_moe 1 \
  --apply_router_aux_loss 1 \
  --router_aux_loss_factor 0.02 \
  --train_epochs 15 \
  --patience 3 \
  --num_workers 4
done

for f in 0.003 0.004 0.005 0.006
do 
accelerate launch --config_file accelerrate_config.yaml --main_process_port 10097 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ../dataset/m4 \
  --model ATM \
  --data m4 \
  --seasonal_patterns Monthly \
  --features M \
  --factor 3 \
  --enc_in 1 \
  --itr 1 \
  --d_model 32 \
  --inter_dim 32 \
  --n_heads 8 \
  --n_querys 8 \
  --prob_bias 0.00 \
  --prob_bias_end 70 \
  --high_freq 1.5 \
  --threshold_ratio 0.991 \
  --aux_loss 1 \
  --aux_loss_factor 0.2 \
  --dropout 0.05 \
  --head_dropout 0.1 \
  --batch_size 36 \
  --learning_rate $f \
  --conv_layers 2 \
  --d_layers 1 \
  --num_experts 8 \
  --top_k 2 \
  --use_moe 1 \
  --apply_router_aux_loss 1 \
  --router_aux_loss_factor 0.02 \
  --train_epochs 15 \
  --patience 3 \
  --num_workers 4
done

for d in 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
do
accelerate launch --config_file accelerrate_config.yaml --main_process_port 10097 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ../dataset/m4 \
  --model ATM \
  --data m4 \
  --seasonal_patterns Weekly \
  --features M \
  --factor 3 \
  --enc_in 1 \
  --itr 1 \
  --d_model 32 \
  --inter_dim 32 \
  --n_heads 8 \
  --n_querys 6 \
  --prob_bias 0.00 \
  --prob_bias_end 70 \
  --high_freq 1.5 \
  --threshold_ratio 0.991 \
  --aux_loss 1 \
  --aux_loss_factor 0.2 \
  --dropout 0.4 \
  --head_dropout 0.1 \
  --batch_size 36 \
  --learning_rate 0.006 \
  --conv_layers 1 \
  --d_layers 2 \
  --num_experts 8 \
  --top_k 2 \
  --use_moe 1 \
  --apply_router_aux_loss 1 \
  --router_aux_loss_factor 0.02 \
  --train_epochs 15 \
  --patience 5 \
  --num_workers 4
done

accelerate launch --config_file accelerrate_config.yaml --main_process_port 10097 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ../dataset/m4 \
  --model ATM \
  --data m4 \
  --seasonal_patterns Daily \
  --features M \
  --factor 3 \
  --enc_in 1 \
  --itr 1 \
  --d_model 32 \
  --inter_dim 32 \
  --n_heads 8 \
  --n_querys 6 \
  --prob_bias 0.00 \
  --prob_bias_end 70 \
  --high_freq 1.5 \
  --threshold_ratio 0.991 \
  --aux_loss 1 \
  --aux_loss_factor 0.2 \
  --dropout 0.05 \
  --head_dropout 0.1 \
  --batch_size 36 \
  --learning_rate 0.006 \
  --conv_layers 1 \
  --d_layers 2 \
  --num_experts 8 \
  --top_k 2 \
  --use_moe 1 \
  --apply_router_aux_loss 1 \
  --router_aux_loss_factor 0.02 \
  --train_epochs 15 \
  --patience 3 \
  --num_workers 4

0.001 0.002 0.003 0.004 0.005 0.006 0.007 0.008 0.009 0.01

for d in  0.0055 0.0057 0.0059 0.0061 0.0063 0.0065
do
accelerate launch --config_file accelerrate_config.yaml --main_process_port 10097 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ../dataset/m4 \
  --model ATM \
  --data m4 \
  --seasonal_patterns Daily \
  --features M \
  --factor 3 \
  --enc_in 1 \
  --itr 1 \
  --d_model 32 \
  --inter_dim 32 \
  --n_heads 8 \
  --n_querys 6 \
  --prob_bias 0.00 \
  --prob_bias_end 70 \
  --high_freq 1.5 \
  --threshold_ratio 0.991 \
  --aux_loss 1 \
  --aux_loss_factor 0.2 \
  --dropout 0.05 \
  --head_dropout 0.1 \
  --batch_size 36 \
  --learning_rate $d \
  --conv_layers 1 \
  --d_layers 2 \
  --num_experts 8 \
  --top_k 2 \
  --use_moe 1 \
  --apply_router_aux_loss 1 \
  --router_aux_loss_factor 0.02 \
  --train_epochs 15 \
  --patience 3 \
  --num_workers 4
done

0.05 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5

for d in 0.05 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5
do
accelerate launch --config_file accelerrate_config.yaml --main_process_port 10097 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ../dataset/m4 \
  --model ATM \
  --data m4 \
  --seasonal_patterns Daily \
  --features M \
  --factor 3 \
  --enc_in 1 \
  --itr 1 \
  --d_model 32 \
  --inter_dim 32 \
  --n_heads 8 \
  --n_querys 6 \
  --prob_bias 0.00 \
  --prob_bias_end 70 \
  --high_freq 1.5 \
  --threshold_ratio 0.991 \
  --aux_loss 1 \
  --aux_loss_factor $d \
  --dropout 0.2 \
  --head_dropout 0.1 \
  --batch_size 36 \
  --learning_rate 0.0025 \
  --conv_layers 2 \
  --d_layers 2 \
  --num_experts 8 \
  --top_k 2 \
  --use_moe 1 \
  --apply_router_aux_loss 1 \
  --router_aux_loss_factor 0.02 \
  --train_epochs 15 \
  --patience 3 \
  --num_workers 4
done

for d in 0.5 1 2 2.5 3 3.5 4 4.5 5
do
accelerate launch --config_file accelerrate_config.yaml --main_process_port 10097 run_m4.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ../dataset/m4 \
  --model ATM \
  --data m4 \
  --seasonal_patterns Hourly \
  --features M \
  --factor 3 \
  --enc_in 1 \
  --itr 1 \
  --d_model 32 \
  --inter_dim 32 \
  --n_heads 8 \
  --n_querys 6 \
  --prob_bias 0.00 \
  --prob_bias_end 70 \
  --high_freq 1.5 \
  --threshold_ratio 0.899 \
  --aux_loss 1 \
  --aux_loss_factor 0.4 \
  --dropout 0.3 \
  --head_dropout 0.1 \
  --batch_size 36 \
  --learning_rate 0.00045 \
  --conv_layers 4 \
  --d_layers 1 \
  --num_experts 8 \
  --top_k 2 \
  --use_moe 1 \
  --apply_router_aux_loss 1 \
  --router_aux_loss_factor 0.02 \
  --train_epochs 15 \
  --patience 3 \
  --num_workers 4
done








