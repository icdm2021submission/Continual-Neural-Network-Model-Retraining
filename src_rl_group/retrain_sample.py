import os
import argparse, time

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0',
                    help="gpu")
parser.add_argument('--loss_fn', default='cnn',
                    help="retrain loss_fn")
parser.add_argument('--use_kfac', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="usek fac true gradient")
parser.add_argument('--log', default='',
                    help="retrain loss_fn")
parser.add_argument('--data_dir', default='../data/mnist',
                    help="Directory containing the dataset")
parser.add_argument('--rl', default='EXP3',
                    help="rl algorithm")
args = parser.parse_args()

model_dir = 'experiments/base_model'
train_log = os.path.join(model_dir, 'train_{}.log'.format(args.log))
test_log = os.path.join(model_dir, 'test_{}.log'.format(args.log))

sample_temp_tf = os.path.join(args.data_dir, 'sample-temp.tfrecords')
sample_tf = os.path.join(args.data_dir, 'sample.tfrecords')

for i in range(5, 10):
	# there was a bug because of reading and writing the same file
	# fixing it using renaming
	os.system('rm corr_weights*')
	time.sleep(200)
	os.system('mv {} {}'.format(sample_temp_tf, sample_tf))	
	train_script = 'CUDA_VISIBLE_DEVICES={} python main.py --train_range {} \
	 --loss_fn {} --finetune true --use_kfac {} --log _{} --data_dir {} --rl {}'.format(args.gpu, i, args.loss_fn, \
	 	args.use_kfac, args.log, args.data_dir, args.rl)
	os.system('echo {} >> {}'.format(train_script, train_log))
	os.system(train_script)
	test_fake_script = 'python evaluate.py --train_range {} \
	 --loss_fn {} --finetune true --use_kfac {} --log _{} --data_dir {}'.format(i, args.loss_fn, \
	 	args.use_kfac, args.log, args.data_dir)
	os.system('echo {} >> {}'.format(test_fake_script, test_log))
	test_script = 'CUDA_VISIBLE_DEVICES={} python evaluate.py \
	 --loss_fn {} --finetune true --use_kfac {} --log _{} --data_dir {}'.format(args.gpu, args.loss_fn, \
	 args.use_kfac, args.log, args.data_dir)	
	os.system(test_script)
	os.system(test_script)	
	if args.use_kfac or args.loss_fn == 'retrain_regu_mine3':
		continue
	collect_script = 'CUDA_VISIBLE_DEVICES={} python collect.py --train_range {} \
	 --loss_fn cnn --log _{} --data_dir {}'.format(args.gpu, i, args.log, args.data_dir)
	os.system(collect_script)
	cluster_script = 'python kmeans.py'
	os.system(cluster_script)
	cluster_script = 'python load_npy.py'
	os.system('python load_npy.py')	
