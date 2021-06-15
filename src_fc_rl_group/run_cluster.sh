rm -r experiments_*
rm *log
./bash.sh

# gpu=$1
# dataset="../data/${2}"

gpu=0
dataset="../data/sea"

mab=EI2
rm "${dataset}/sample-temp.tfrecords"
rm "${dataset}/sample.tfrecords"

CUDA_VISIBLE_DEVICES=$gpu python main.py --train_range [0-4] --data_dir $dataset --rl $mab
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --train_range [0-4] --data_dir $dataset
python kmeans.py
python load_npy.py
cp -r experiments experiments_base
cp -r weights weights_base

CUDA_VISIBLE_DEVICES=$gpu python collect.py --train_range [0-4] --data_dir $dataset
cp -r experiments experiments_collect
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --train_range [0-4] --data_dir $dataset

