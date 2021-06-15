import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/sea',
                    help="Directory containing the dataset")
parser.add_argument('--gpu', default='0',
                    help="gpu")
parser.add_argument('--rl', default='EI2',
                    help="rl algorithm")
args = parser.parse_args()

loss_fns = ['retrain_regu_mine']
# loss_fns = ['retrain_regu_mas', 'retrain_regu_mine', 'retrain_regu_minen', 'retrain_regu_fisher', 'retrain_regu_fishern',\
# 'retrain', 'retrain_regu', 'retrain_regu_selfless']#'retrain_regu_mine3', 'cnn'

model_log_map = {'fishern': 'EWCN', 'mine': 'Arms', 'minen': 'ArmsN', 'regu': 'RetrainRegu', \
'fisher': 'EWC', 'mas': 'MAS', 'selfless': 'Selfless', 'retrain':'Retrain'}
for loss_fn in loss_fns:
	log = loss_fn.split('_')[-1]
	log_loss_fn = model_log_map[log]
	if loss_fn == 'cnn':
		log_loss_fn = 'kfac'
	# sample_temp_tf = os.path.join(args.data_dir, 'sample-temp.tfrecords')
	# sample_tf = os.path.join(args.data_dir, 'sample.tfrecords')
	# # there was a bug because of reading and writing the same file
	# # fixing it using renaming
	# os.system('mv {} {}'.format(sample_temp_tf, sample_tf))
	for cluster in range(2, 5):
		os.system('mv experiments/base_model/*.log ./')
		os.system('rm -rf experiments')
		os.system('cp -r experiments_collect experiments')
		os.system('rm -rf weights')
		os.system('cp -r weights_base weights')

		cluster_script = 'python kmeans.py {}'.format(cluster)
		os.system(cluster_script)
		cluster_script = 'python load_npy.py'
		os.system(cluster_script)

		log = '{}_{}'.format(log_loss_fn, cluster)
		model_dir = 'experiments/base_model'
		train_log = os.path.join(model_dir, 'train_{}.log'.format(log))
		test_log = os.path.join(model_dir, 'test_{}.log'.format(log))
		train_script= 'CUDA_VISIBLE_DEVICES={} python main.py --train_range {} \
		--loss_fn {} --finetune true --log _{} --data_dir {} --rl {} \
		--num_clusters {}'.format(args.gpu, 5, loss_fn, log, args.data_dir, args.rl, cluster)
		if loss_fn == 'cnn':
			train_script += ' --use_kfac true'
		print(train_script)
		os.system(train_script)
		os.system('echo {} >> {}'.format(train_script, train_log))

		test_fake_script = 'python evaluate.py --train_range {} \
		 --loss_fn {} --finetune true --log _{} --data_dir {}'.format(5, loss_fn, \
		 	log, args.data_dir)
		os.system('echo {} >> {}'.format(test_fake_script, test_log))
		test_script = 'CUDA_VISIBLE_DEVICES={} python evaluate.py --train_range [0-{}] \
		 --loss_fn {} --finetune true --log _{} --data_dir {}'.format(args.gpu, 5, loss_fn, \
		 log, args.data_dir)	
		os.system(test_script)
		os.system('echo {} >> {}'.format(test_script, test_log))

		# os.system('mv experiments/base_model/*.log ./')
		# os.system('rm -rf experiments')
		# os.system('cp -r experiments_collect experiments')
		# os.system('rm -rf weights')
		# os.system('cp -r weights_base weights')		
os.system('mv experiments/base_model/*.log ./')
