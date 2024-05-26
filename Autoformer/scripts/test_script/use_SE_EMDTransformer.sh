#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path BTP.csv \
  --model_id EnvFormer \
  --model SE_EMDTransformer \
  --data custom \
  --features MS \
  --validate_step 10 \
  --seq_len 10 \
  --label_len 10 \
  --pred_len 10 \
  --moving_avg 21 \
  --batch_size 64 \
  --d_model 64 \
  --n_heads 12 \
  --patience 4 \
  --freq s \
  --learning_rate 0.0009 \
  --weight_decay 0 \
  --train_epochs 16 \
  --d_ff 256 \
  --e_layers 1 \
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
#   --data_path BTP_May.csv \
#   --model_id May \
#   --model SE_EMDTransformer \
#   --data custom \
#   --features MS \
#   --validate_step 10 \
#   --seq_len 20 \
#   --label_len 10 \
#   --pred_len 10 \
#   --moving_avg 21 \
#   --batch_size 64 \
#   --d_model 74 \
#   --n_heads 12 \
#   --patience 3 \
#   --freq s \
#   --learning_rate 0.0009 \
#   --weight_decay 0.0001 \
#   --train_epochs 16 \
#   --d_ff 320 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP_May.csv \
#   --model_id May \
#   --model SE_EMDTransformer \
#   --data custom \
#   --features MS \
#   --validate_step 10 \
#   --seq_len 20 \
#   --label_len 10 \
#   --pred_len 10 \
#   --moving_avg 21 \
#   --batch_size 64 \
#   --d_model 74 \
#   --n_heads 12 \
#   --patience 3 \
#   --freq s \
#   --learning_rate 0.0005 \
#   --weight_decay 0.00001 \
#   --train_epochs 16 \
#   --d_ff 320 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP_May.csv \
#   --model_id May \
#   --model SE_EMDTransformer \
#   --data custom \
#   --features MS \
#   --validate_step 10 \
#   --seq_len 20 \
#   --label_len 10 \
#   --pred_len 10 \
#   --moving_avg 21 \
#   --batch_size 64 \
#   --d_model 74 \
#   --n_heads 12 \
#   --patience 3 \
#   --freq s \
#   --learning_rate 0.0004 \
#   --weight_decay 0.00001 \
#   --train_epochs 16 \
#   --d_ff 320 \
#   --e_layers 1 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1




  

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_1 \
#   --model STDTransformer \
#   --data custom \
#   --features MS \
#   --validate_step 5 \
#   --seq_len 10 \
#   --label_len 5 \
#   --pred_len 5 \
#   --moving_avg 11 \
#   --d_model 720 \
#   --n_heads 12 \
#   --patience 7 \
#   --freq m \
#   --learning_rate 0.0001 \
#   --weight_decay 5 \
#   --train_epochs 25 \
#   --d_ff 1000 \
#   --e_layers 6 \
#   --d_layers 4 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_1 \
#   --model STDTransformer \
#   --data custom \
#   --features MS \
#   --validate_step 5 \
#   --seq_len 10 \
#   --label_len 5 \
#   --pred_len 5 \
#   --moving_avg 11 \
#   --d_model 960 \
#   --n_heads 12 \
#   --patience 7 \
#   --freq m \
#   --learning_rate 0.0001 \
#   --weight_decay 5 \
#   --train_epochs 25 \
#   --d_ff 1000 \
#   --e_layers 6 \
#   --d_layers 4 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_1 \
#   --model STDTransformer \
#   --data custom \
#   --features MS \
#   --validate_step 5 \
#   --seq_len 10 \
#   --label_len 5 \
#   --pred_len 5 \
#   --moving_avg 11 \
#   --d_model 640 \
#   --n_heads 12 \
#   --patience 7 \
#   --freq m \
#   --learning_rate 0.0001 \
#   --weight_decay 6 \
#   --train_epochs 25 \
#   --d_ff 1000 \
#   --e_layers 6 \
#   --d_layers 4 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1

#  python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_1 \
#   --model STDTransformer \
#   --data custom \
#   --features MS \
#   --validate_step 5 \
#   --seq_len 10 \
#   --label_len 5 \
#   --pred_len 5 \
#   --moving_avg 11 \
#   --d_model 512 \
#   --n_heads 14 \
#   --patience 7 \
#   --freq m \
#   --learning_rate 0.0001 \
#   --weight_decay 6 \
#   --train_epochs 25 \
#   --d_ff 1000 \
#   --e_layers 6 \
#   --d_layers 4 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1 