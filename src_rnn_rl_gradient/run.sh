# rm -r experiments_*
# rm *log
# ./bash.sh
# # ./run.sh 1 mnist EI
# gpu=$1
# dataset="../data/${2}"
# mab=$3

# rm "${dataset}/sample-temp.tfrecords"
# rm "${dataset}/sample.tfrecords"

# CUDA_VISIBLE_DEVICES=$gpu python main.py --train_range [0-4] --data_dir $dataset --rl $mab
# CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset
# cp -r experiments experiments_base

# CUDA_VISIBLE_DEVICES=$gpu python collect.py --train_range [0-4] --data_dir $dataset
# cp -r experiments experiments_collect
# CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset

# python runs.py --gpu $gpu --data_dir $dataset --rl EI
# rm -r "result_${2}_${3}"
# mkdir "result_${2}_${3}"
# mv *.log "result_${2}_${3}"
#--------------------------------------------------------------------------------------------
# ./run.sh 1 mnist
gpu=$1
dataset="../data/${2}"
#-------------------Greedy
mab=Greedy

rm -r experiments_*
rm *log
./bash.sh

rm "${dataset}/sample-temp.tfrecords"
rm "${dataset}/sample.tfrecords"

CUDA_VISIBLE_DEVICES=$gpu python main.py --train_range [0-4] --data_dir $dataset --rl $mab
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset
cp -r experiments experiments_base

CUDA_VISIBLE_DEVICES=$gpu python collect.py --train_range [0-4] --data_dir $dataset
cp -r experiments experiments_collect
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset

python runs.py --gpu $gpu --data_dir $dataset --rl $mab
rm -r "result_${2}_${mab}"
mkdir "result_${2}_${mab}"
mv *.log "result_${2}_${mab}"
#-------------------Reservoir
mab=Reservoir

rm -r experiments_*
rm *log
./bash.sh

rm "${dataset}/sample-temp.tfrecords"
rm "${dataset}/sample.tfrecords"

CUDA_VISIBLE_DEVICES=$gpu python main.py --train_range [0-4] --data_dir $dataset --rl $mab
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset
cp -r experiments experiments_base

CUDA_VISIBLE_DEVICES=$gpu python collect.py --train_range [0-4] --data_dir $dataset
cp -r experiments experiments_collect
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset

python runs.py --gpu $gpu --data_dir $dataset --rl $mab
rm -r "result_${2}_${mab}"
mkdir "result_${2}_${mab}"
mv *.log "result_${2}_${mab}"
# -------------------EI
mab=EI

rm -r experiments_*
rm *log
./bash.sh

rm "${dataset}/sample-temp.tfrecords"
rm "${dataset}/sample.tfrecords"

CUDA_VISIBLE_DEVICES=$gpu python main.py --train_range [0-4] --data_dir $dataset --rl $mab
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset
cp -r experiments experiments_base

CUDA_VISIBLE_DEVICES=$gpu python collect.py --train_range [0-4] --data_dir $dataset
cp -r experiments experiments_collect
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset

python runs.py --gpu $gpu --data_dir $dataset --rl $mab
rm -r "result_${2}_${mab}"
mkdir "result_${2}_${mab}"
mv *.log "result_${2}_${mab}"

# -------------------EI2
mab=EI2

rm -r experiments_*
rm *log
./bash.sh

rm "${dataset}/sample-temp.tfrecords"
rm "${dataset}/sample.tfrecords"

CUDA_VISIBLE_DEVICES=$gpu python main.py --train_range [0-4] --data_dir $dataset --rl $mab
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset
cp -r experiments experiments_base

CUDA_VISIBLE_DEVICES=$gpu python collect.py --train_range [0-4] --data_dir $dataset
cp -r experiments experiments_collect
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset

python runs.py --gpu $gpu --data_dir $dataset --rl $mab
rm -r "result_${2}_${mab}"
mkdir "result_${2}_${mab}"
mv *.log "result_${2}_${mab}"

#-------------------EXP3
mab=EXP3

rm -r experiments_*
rm *log
./bash.sh

rm "${dataset}/sample-temp.tfrecords"
rm "${dataset}/sample.tfrecords"

CUDA_VISIBLE_DEVICES=$gpu python main.py --train_range [0-4] --data_dir $dataset --rl $mab
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset
cp -r experiments experiments_base

CUDA_VISIBLE_DEVICES=$gpu python collect.py --train_range [0-4] --data_dir $dataset
cp -r experiments experiments_collect
CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset

python runs.py --gpu $gpu --data_dir $dataset --rl $mab
rm -r "result_${2}_${mab}"
mkdir "result_${2}_${mab}"
mv *.log "result_${2}_${mab}"

# #-------------------UCB
# mab=UCB

# rm -r experiments_*
# rm *log
# ./bash.sh

# rm "${dataset}/sample-temp.tfrecords"
# rm "${dataset}/sample.tfrecords"

# CUDA_VISIBLE_DEVICES=$gpu python main.py --train_range [0-4] --data_dir $dataset --rl $mab
# CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset
# cp -r experiments experiments_base

# CUDA_VISIBLE_DEVICES=$gpu python collect.py --train_range [0-4] --data_dir $dataset
# cp -r experiments experiments_collect
# CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset

# python runs.py --gpu $gpu --data_dir $dataset --rl $mab
# rm -r "result_${2}_${mab}"
# mkdir "result_${2}_${mab}"
# mv *.log "result_${2}_${mab}"

# #-------------------TS
# mab=TS

# rm -r experiments_*
# rm *log
# ./bash.sh

# rm "${dataset}/sample-temp.tfrecords"
# rm "${dataset}/sample.tfrecords"

# CUDA_VISIBLE_DEVICES=$gpu python main.py --train_range [0-4] --data_dir $dataset --rl $mab
# CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset
# cp -r experiments experiments_base

# CUDA_VISIBLE_DEVICES=$gpu python collect.py --train_range [0-4] --data_dir $dataset
# cp -r experiments experiments_collect
# CUDA_VISIBLE_DEVICES=$gpu python evaluate.py --data_dir $dataset

# python runs.py --gpu $gpu --data_dir $dataset --rl $mab
# rm -r "result_${2}_${mab}"
# mkdir "result_${2}_${mab}"
# mv *.log "result_${2}_${mab}"


# python runs.py --gpu $gpu --data_dir $dataset --rl EI2
# rm -r "result_${2}_EI2"
# mkdir "result_${2}_EI2"
# mv *.log "result_${2}_EI2"

# #--------------------------------------------------------------------------------------------
# # python runs.py --gpu $gpu --data_dir $dataset --rl UCB
# # rm -r "result_${2}_UCB"
# # mkdir "result_${2}_UCB"
# # mv *.log "result_${2}_UCB"

# # python runs.py --gpu $gpu --data_dir $dataset --rl TS
# # rm -r "result_${2}_TS"
# # mkdir "result_${2}_TS"
# # mv *.log "result_${2}_TS"

# # python runs.py --gpu $gpu --data_dir $dataset --rl EXP3
# # rm -r "result_${2}_EXP3"
# # mkdir "result_${2}_EXP3"
# # mv *.log "result_${2}_EXP3"