export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path BTP.csv \
  --model_id test_50_5 \
  --model Single_BiLSTM \
  --data custom \
  --features MS \
  --seq_len 30 \
  --label_len 0 \
  --pred_len 10 \
  --validate_step 10 \
  --rnn_layers 1 \
  --factor 3 \
  --d_model 74 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 37 \
  --des 'Exp' \
  --itr 1