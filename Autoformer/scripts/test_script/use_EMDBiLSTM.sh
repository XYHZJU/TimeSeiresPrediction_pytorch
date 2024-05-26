export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path EMD.csv \
  --model_id EMD_test \
  --model EMDBiLSTM \
  --data custom \
  --features MS \
  --seq_len 30 \
  --label_len 0 \
  --pred_len 10 \
  --batch_size 64 \
  --validate_step 10 \
  --rnn_layers 3 \
  --factor 3 \
  --d_model 74 \
  --enc_in 17 \
  --dec_in 17 \
  --c_out 17 \
  --des 'Exp' \
  --itr 1