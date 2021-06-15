import os
import argparse
from static import loss_fns, model_log_map

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0',
                    help="gpu")
parser.add_argument('--data_dir', default='../data/imdb',
                    help="Directory containing the dataset")

# loss_fns = ['retrain_regu_mine', 'retrain_regu_fisher', 'retrain_regu_fishern'\
# 'retrain_regu_mas', 'retrain', 'retrain_regu', 'retrain_regu_selfless']#'retrain_regu_mine3', 'cnn'
args = parser.parse_args()

for loss_fn in loss_fns:
	log = loss_fn.split('_')[-1]
	log = model_log_map[log]
	if loss_fn == 'cnn':
		log = 'kfac'
	script= 'python retrain.py --loss_fn {} --log {} --gpu {} --data_dir {}'.format(loss_fn, \
		log, args.gpu, args.data_dir)
	if loss_fn == 'cnn':
		script += ' --use_kfac true'
	os.system(script)
	os.system('mv experiments/base_model/*.log ./')
	os.system('rm -rf experiments')
	if loss_fn == 'retrain_regu_mine3':
		os.system('cp -r experiments_base experiments')
	else:
		os.system('cp -r experiments_collect experiments')
os.system('mv experiments/base_model/*.log ./')

