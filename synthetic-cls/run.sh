#!/bin/bash

CUDA_VISIBLE_DEVICES=0 ./icnn.py --dataset linear  --model picnn --nEpoch 200 &
CUDA_VISIBLE_DEVICES=1 ./icnn.py --dataset moons   --model picnn --nEpoch 200 &
CUDA_VISIBLE_DEVICES=2 ./icnn.py --dataset circles --model picnn --nEpoch 200 &

CUDA_VISIBLE_DEVICES=3 ./icnn.py --dataset linear  --model ficnn --nEpoch 200 &
CUDA_VISIBLE_DEVICES=0 ./icnn.py --dataset moons   --model ficnn --nEpoch 200 &
CUDA_VISIBLE_DEVICES=1 ./icnn.py --dataset circles --model ficnn --nEpoch 200 &

wait

./make-tile.sh
