import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/reuters',
                    help="Directory containing the dataset")
parser.add_argument('--gpu', default='0',
                    help="gpu")
parser.add_argument('--rl', default='EXP3',
                    help="rl algorithm")
args = parser.parse_args()

loss_fns = ['retrain_regu_mine', 'retrain_regu_minen']
# loss_fns = ['retrain_regu_mas', 'retrain_regu_mine', 'retrain_regu_minen', 'retrain_regu_fisher', 'retrain_regu_fishern',\
# 'retrain', 'retrain_regu', 'retrain_regu_selfless']#'retrain_regu_mine3', 'cnn'

model_log_map = {'fishern': 'EWCN','mine': 'Arms', 'minen': 'ArmsN', 'regu': 'RetrainRegu', \
'fisher': 'EWC', 'mas': 'MAS', 'selfless': 'Selfless', 'retrain':'Retrain'}
for loss_fn in loss_fns:
	log = loss_fn.split('_')[-1]
	log = model_log_map[log]
	if loss_fn == 'cnn':
		log = 'kfac'
	# sample_temp_tf = os.path.join(args.data_dir, 'sample-temp.tfrecords')
	# sample_tf = os.path.join(args.data_dir, 'sample.tfrecords')
	# # there was a bug because of reading and writing the same file
	# # fixing it using renaming
	# os.system('mv {} {}'.format(sample_temp_tf, sample_tf))
	script= 'python retrain_sample.py --loss_fn {} --log {}_s --gpu {} --data_dir {} --rl {}'.format(loss_fn, \
		log, args.gpu, args.data_dir, args.rl)
	if loss_fn == 'cnn':
		script += ' --use_kfac true'
	os.system(script)
	os.system('mv experiments/base_model/*.log ./')
	os.system('rm -rf experiments')
	os.system('cp -r experiments_collect experiments')
os.system('mv experiments/base_model/*.log ./')
