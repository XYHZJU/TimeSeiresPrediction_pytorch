#!/bin/bash
export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path test2.csv \
  --model_id test_96_96 \
  --model LogTrans \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 20 \
  --dec_in 20 \
  --c_out 20 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path test2.csv \
  --model_id test_96_192 \
  --model LogTrans \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 20 \
  --dec_in 20 \
  --c_out 20 \
  --des 'Exp' \
  --itr 1

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path test2.csv \
  --model_id test_96_336 \
  --model LogTrans \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 20 \
  --dec_in 20 \
  --c_out 20 \
  --des 'Exp' \
  --itr 1 \


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path test2.csv \
  --model_id test_96_720 \
  --model LogTrans \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 20 \
  --dec_in 20 \
  --c_out 20 \
  --des 'Exp' \
  --itr 1