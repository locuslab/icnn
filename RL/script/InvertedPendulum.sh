N=10
total=10000
trainTimestep=100
testEpisode=1
task="InvertedPendulum-v1"
outdir="output/InvertedPendulum-v1/"
monitor=-1

#DDPG
for ((i=0;i<$N;i++))
do
  python src/main.py --model DDPG --env $task --outdir $outdir/DDPG/$i \
    --total $total --train $trainTimestep --test $testEpisode --reward_k 0.3 \
    --tfseed $i --gymseed $i --npseed $i --monitor $monitor
done

#NAF
for ((i=0;i<$N;i++))
do
  python src/main.py --model NAF --env $task --outdir $outdir/NAF/$i \
    --total $total --train $trainTimestep --test $testEpisode --reward_k 0.2 \
    --tfseed $i --gymseed $i --npseed $i --monitor $monitor
done

#ICNN
for ((i=0;i<$N;i++))
do
  python src/main.py --model ICNN --env $task --outdir $outdir/ICNN/$i \
    --total $total --train $trainTimestep --test $testEpisode --reward_k 0.3\
    --tfseed $i --gymseed $i --npseed $i --monitor $monitor
done

#plot
python src/plot.py --runs $N --total $total --train $trainTimestep --data $outdir --min 0 --max 1100
