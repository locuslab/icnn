#!/bin/bash

set -x -e
cd $(dirname $0)/..

N=10
total=100000
trainTimestep=1000
testEpisode=1
task="HalfCheetah-v1"
outdir="output/HalfCheetah-v1/"
monitor=-1

# DDPG
for ((i=0;i<$N;i++))
do
  python3 src/main.py --model DDPG --env $task --outdir $outdir/DDPG/$i \
    --total $total --train $trainTimestep --test $testEpisode \
    --tfseed $i --gymseed $i --npseed $i --monitor $monitor --l2norm 0.01
done


#NAF
for ((i=0;i<$N;i++))
do
  python3 src/main.py --model NAF --env $task --outdir $outdir/NAF/$i \
    --total $total --train $trainTimestep --test $testEpisode --reward_k 0.1\
    --tfseed $i --gymseed $i --npseed $i  --monitor $monitor \
    --initstd 0.007 --naf_bn True
done

#ICNN
for ((i=0;i<$N;i++))
do
  python3 src/main.py --model ICNN --env $task --outdir $outdir/ICNN/$i \
    --total $total --train $trainTimestep --test $testEpisode --reward_k 0.4\
    --tfseed $i --gymseed $i --npseed $i --monitor $monitor --l2norm 0.001 --icnn_bn True
done

python3 src/plot.py --ymin -500 --ymax 5500 $outdir
