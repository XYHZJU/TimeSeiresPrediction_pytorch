export CUDA_VISIBLE_DEVICES=0


python -u run.py \
  --is_training 1 \
  --root_path ./dataset/datagen/ \
  --data_path test2.csv \
  --model_id test_50_1 \
  --model MultiRNN \
  --data custom \
  --features MS \
  --seq_len 50 \
  --label_len 10 \
  --pred_len 1 \
  --rnn_layers 2 \
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
  --model_id test_50_2 \
  --model MultiRNN \
  --data custom \
  --features MS \
  --seq_len 50 \
  --label_len 10 \
  --pred_len 2 \
  --rnn_layers 2 \
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
  --model_id test_50_3 \
  --model MultiRNN \
  --data custom \
  --features MS \
  --seq_len 50 \
  --label_len 10 \
  --pred_len 3 \
  --rnn_layers 2 \
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
  --model_id test_50_4 \
  --model MultiRNN \
  --data custom \
  --features MS \
  --seq_len 50 \
  --label_len 10 \
  --pred_len 4 \
  --rnn_layers 2 \
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
  --model_id test_50_5 \
  --model MultiRNN \
  --data custom \
  --features MS \
  --seq_len 50 \
  --label_len 10 \
  --pred_len 5 \
  --rnn_layers 2 \
  --factor 3 \
  --enc_in 20 \
  --dec_in 20 \
  --c_out 20 \
  --des 'Exp' \
  --itr 1``