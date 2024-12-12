#!/bin/bash
# python main.py --seed 2020 --dataset "ratio_0.6" --data_path "./Data/" --epochs 120 --batch_size 16 --GCNLayer 2 --agg "sum" --test_batch_size 4 --l2 0.0001 --lr 0.001 --topK "[5, 10, 15]" --dim 16
# python main.py --seed 2020 --dataset "ratio_0.7" --data_path "./Data/" --epochs 120 --batch_size 16 --GCNLayer 2 --agg "sum" --test_batch_size 4 --l2 0.0001 --lr 0.001 --topK "[5, 10, 15]" --dim 16
# python main.py --seed 2020 --dataset "ratio_0.8" --data_path "./Data/" --epochs 120 --batch_size 16 --GCNLayer 2 --agg "sum" --test_batch_size 4 --l2 0.0001 --lr 0.001 --topK "[5, 10, 15]" --dim 16
# python main.py --seed 2020 --dataset "ratio_0.9" --data_path "./Data/" --epochs 120 --batch_size 16 --GCNLayer 2 --agg "sum" --test_batch_size 4 --l2 0.0001 --lr 0.001 --topK "[5, 10, 15]" --dim 16
# python main.py --seed 2020 --dataset "ratio_1.0" --data_path "./Data/" --epochs 120 --batch_size 16 --GCNLayer 2 --agg "sum" --test_batch_size 4 --l2 0.0001 --lr 0.001 --topK "[5, 10, 15]" --dim 16


python eval.py --seed 2020 --dataset "ratio_0.6" --data_path "./Data/" --epochs 120 --batch_size 16 --GCNLayer 2 --agg "sum" --test_batch_size 4 --l2 0.0001 --lr 0.001 --topK "[5, 10, 15]" --dim 16
python eval.py --seed 2020 --dataset "ratio_0.7" --data_path "./Data/" --epochs 120 --batch_size 16 --GCNLayer 2 --agg "sum" --test_batch_size 4 --l2 0.0001 --lr 0.001 --topK "[5, 10, 15]" --dim 16
python eval.py --seed 2020 --dataset "ratio_0.8" --data_path "./Data/" --epochs 120 --batch_size 16 --GCNLayer 2 --agg "sum" --test_batch_size 4 --l2 0.0001 --lr 0.001 --topK "[5, 10, 15]" --dim 16
python eval.py --seed 2020 --dataset "ratio_0.9" --data_path "./Data/" --epochs 120 --batch_size 16 --GCNLayer 2 --agg "sum" --test_batch_size 4 --l2 0.0001 --lr 0.001 --topK "[5, 10, 15]" --dim 16
python eval.py --seed 2020 --dataset "ratio_1.0" --data_path "./Data/" --epochs 120 --batch_size 16 --GCNLayer 2 --agg "sum" --test_batch_size 4 --l2 0.0001 --lr 0.001 --topK "[5, 10, 15]" --dim 16
