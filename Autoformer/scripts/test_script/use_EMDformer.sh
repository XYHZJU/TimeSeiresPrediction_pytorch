export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path BTP.csv \
  --model_id BTP_100_1 \
  --model EMDformer \
  --data custom \
  --features MS \
  --seq_len 10 \
  --label_len 10 \
  --pred_len 10 \
  --batch_size 128 \
  --validate_step 10 \
  --learning_rate 0.0009 \
  --weight_decay 8 \
  --d_ff 1024 \
  --d_model 256 \
  --e_layers 3 \
  --d_layers 2 \
  --factor 3 \
  --freq t \
  --moving_avg 15 \
  --patience 5 \
  --enc_in 37 \
  --dec_in 37 \
  --c_out 37 \
  --des 'Exp' \
  --itr 1 \
  --train_epochs 16

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
#   --pred_len 5 \
#   --validate_step 2 \
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