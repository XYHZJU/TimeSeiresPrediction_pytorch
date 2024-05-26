export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path BTP.csv \
  --model_id April+March \
  --model Autoformer \
  --data custom \
  --features MS \
  --seq_len 10 \
  --label_len 10 \
  --pred_len 10 \
  --validate_step 10 \
  --learning_rate 0.0005 \
  --weight_decay 1 \
  --d_ff 256 \
  --d_model 64 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --freq t \
  --moving_avg 25 \
  --patience 7 \
  --enc_in 37 \
  --dec_in 37 \
  --c_out 37 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 16

# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP0.csv \
#   --model_id BTP_100_1 \
#   --model Autoformer \
#   --data custom \
#   --features MS \
#   --seq_len 10 \
#   --label_len 5 \
#   --pred_len 5 \
#   --validate_step 5 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --patience 3 \
#   --d_ff 256 \
#   --d_model 64 \
#   --moving_avg 25 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1 \


# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_3 \
#   --model Autoformer \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 10 \
#   --pred_len 5 \
#   --validate_step 3 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1 \


# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_4 \
#   --model Autoformer \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 10 \
#   --pred_len 5 \
#   --validate_step 4 \
#   --e_layers 2 \
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
#   --model_id BTP_100_5 \
#   --model Autoformer \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 10 \
#   --pred_len 5 \
#   --validate_step 5 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1

#     python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_1 \
#   --model Autoformer \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 10 \
#   --pred_len 10 \
#   --validate_step 6 \
#   --e_layers 2 \
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
#   --model_id BTP_100_2 \
#   --model Autoformer \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 10 \
#   --pred_len 10 \
#   --validate_step 7 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1 \


# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_3 \
#   --model Autoformer \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 10 \
#   --pred_len 10 \
#   --validate_step 8 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1 \


# python -u run.py \
#   --is_training 1 \
#   --root_path ./dataset/datagen/ \
#   --data_path BTP.csv \
#   --model_id BTP_100_4 \
#   --model Autoformer \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 10 \
#   --pred_len 10 \
#   --validate_step 9 \
#   --e_layers 2 \
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
#   --model_id BTP_100_5 \
#   --model Autoformer \
#   --data custom \
#   --features MS \
#   --seq_len 100 \
#   --label_len 10 \
#   --pred_len 10 \
#   --validate_step 10 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 37 \
#   --dec_in 37 \
#   --c_out 37 \
#   --des 'Exp' \
#   --itr 1