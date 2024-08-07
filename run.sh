python train.py -s data/e050c15a8d --iterations 30000 --data_device cpu --model_path out/e050c15a8d_3 --eval -r 1
python render.py -m  out/e050c15a8d_2 --skip_train
python metrics.py -m  out/e050c15a8d_2