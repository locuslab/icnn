#!/bin/bash

N=10
total=100000
trainTimestep=1000
testEpisode=1
task="Swimmer-v1"
outdir="output/Swimmer-v1/"
monitor=-1

#DDPG
for ((i=0;i<$N;i++))
do
  python3 src/main.py --model DDPG --env $task --outdir $outdir/DDPG/$i \
    --total $total --train $trainTimestep --test $testEpisode \
    --tfseed $i --gymseed $i --npseed $i --monitor $monitor --l2norm 0.001 --ousigma 1.
done

#NAF
for ((i=0;i<$N;i++))
do
  python3 src/main.py --model NAF --env $task --outdir $outdir/NAF/$i \
    --total $total --train $trainTimestep --test $testEpisode \
    --tfseed $i --gymseed $i --npseed $i  --monitor $monitor --l2norm 0.001 --naf_bn True\
    --initstd 0.007 --naf_bn True
done

#ICNN
for ((i=0;i<$N;i++))
do
  python3 src/main.py --model ICNN --env $task --outdir $outdir/ICNN/$i \
    --total $total --train $trainTimestep --test $testEpisode --reward_k 0.3\
    --tfseed $i --gymseed $i --npseed $i --monitor $monitor --l2norm 0.001 --icnn_bn True
done

python3 src/plot.py --ymin -50 --ymax 450 $outdir
