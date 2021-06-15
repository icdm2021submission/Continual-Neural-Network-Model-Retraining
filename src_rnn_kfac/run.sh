gpu=$1
dataset="../data/${2}"

rm -r experiments_*
rm *log
./bash.sh
#../data/mnist
# ./run.sh 1 mnist

# script=$3
echo $dataset
CUDA_VISIBLE_DEVICES=$gpu python main.py --train_range [0-4] --data_dir $dataset
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset
cp -r experiments experiments_base

CUDA_VISIBLE_DEVICES=$gpu python collect.py --train_range [0-4] --data_dir $dataset
cp -r experiments experiments_collect
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset
# cd ../
# cp -r src src_0
# cp -r src src_2
# cd src

python run1.py --gpu $gpu --data_dir $dataset
rm -r "result_${2}_1"
mkdir "result_${2}_1"
mv *_*g1.log "result_${2}_1"

python run2.py --gpu $gpu --data_dir $dataset
rm -r "result_${2}_2"
mkdir "result_${2}_2"
mv *_*g2.log "result_${2}_2"

python run0.py --gpu $gpu --data_dir $dataset
rm -r "result_${2}_0"
mkdir "result_${2}_0"
mv *_*.log "result_${2}_0"