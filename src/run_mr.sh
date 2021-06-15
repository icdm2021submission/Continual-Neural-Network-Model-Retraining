gpu=$1
dataset="../data/${2}"

# rm -r "experiments_${2}*"
rm *log
./bash.sh
#../data/mnist
# ./run.sh 1 mnist

# script=$3
echo $dataset

CUDA_VISIBLE_DEVICES=$gpu python main.py --data_dir $dataset --train_range [0-9] --use_bn false --training_keep_prob 1.0 --log _null
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset --log _null
rm -r "result_${2}_null"
mkdir "result_${2}_null"
mv experiments/base_model/*_*.log "result_${2}_null"
rm -r "experiments_${2}_null"
cp -r experiments "experiments_${2}_null"



CUDA_VISIBLE_DEVICES=$gpu python main.py --data_dir $dataset --train_range [0-9] --use_bn true --training_keep_prob 1.0 --log _bn
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset --use_bn true --log _bn
rm -r "result_${2}_bn"
mkdir "result_${2}_bn"
mv experiments/base_model/*_*.log "result_${2}_bn"
rm -r "experiments_${2}_bn"
cp -r experiments "experiments_${2}_bn"


CUDA_VISIBLE_DEVICES=$gpu python main.py --data_dir $dataset --train_range [0-9] --use_bn false --training_keep_prob 0.7 --log _dr
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset --log _dr
rm -r "result_${2}_dr"
mkdir "result_${2}_dr"
mv experiments/base_model/*_*.log "result_${2}_dr"
rm -r "experiments_${2}_dr"
cp -r experiments "experiments_${2}_dr"


CUDA_VISIBLE_DEVICES=$gpu python main.py --data_dir $dataset --train_range [0-9] --use_bn true --training_keep_prob 0.7 --log _mr
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset --use_bn true --log _mr
rm -r "result_${2}_mr"
mkdir "result_${2}_mr"
mv experiments/base_model/*_*.log "result_${2}_mr"
rm -r "experiments_${2}_mr"
cp -r experiments "experiments_${2}_mr"


CUDA_VISIBLE_DEVICES=$gpu python main.py --data_dir $dataset --train_range [0-9] --use_bn true --training_keep_prob 0.1 --log _mr
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset --use_bn true --log _mr
rm -r "result_${2}_mr1"
mkdir "result_${2}_mr1"
mv experiments/base_model/*_*.log "result_${2}_mr1"
rm -r "experiments_${2}_mr1"
cp -r experiments "experiments_${2}_mr1"