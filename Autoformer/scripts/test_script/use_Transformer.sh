#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path BTP.csv \
  --model_id BTP_100_1 \
  --model Transformer \
  --data custom \
  --features MS \
  --validate_step 10 \
  --seq_len 30 \
  --label_len 10 \
  --pred_len 10 \
  --d_model 88 \
  --n_heads 8 \
  --patience 7 \
  --freq t\
  --learning_rate 0.0009 \
  --weight_decay 4 \
  --train_epochs 20 \
  --d_ff 288 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 37 \
  --dec_in 37 \
  --c_out 37 \
  --des 'Exp' \
  --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path test2.csv \
#   --model_id test_100_2 \
#   --model Transformer \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 0 \
#   --pred_len 2 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 20 \
#   --dec_in 20 \
#   --c_out 20 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path test2.csv \
#   --model_id test_100_3 \
#   --model Transformer \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 0 \
#   --pred_len 3 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 20 \
#   --dec_in 20 \
#   --c_out 20 \
#   --des 'Exp' \
#   --itr 1 \


# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path test2.csv \
#   --model_id test_100_4 \
#   --model Transformer \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 0 \
#   --pred_len 4 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 20 \
#   --dec_in 20 \
#   --c_out 20 \
#   --des 'Exp' \
#   --itr 1

#   python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path test2.csv \
#   --model_id test_100_5 \
#   --model Transformer \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 0 \
#   --pred_len 5 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 20 \
#   --dec_in 20 \
#   --c_out 20 \
#   --des 'Exp' \
#   --itr 1