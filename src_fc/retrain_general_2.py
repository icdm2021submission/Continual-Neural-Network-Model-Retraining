import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0',
                    help="gpu")
parser.add_argument('--loss_fn', default='cnn',
                    help="retrain loss_fn")
parser.add_argument('--use_kfac', default=False, type=lambda x: (str(x).lower() in ['true','1', 'yes']), \
    help="usek fac true gradient")
parser.add_argument('--log', default='',
                    help="retrain loss_fn")
parser.add_argument('--data_dir', default='../data/cifar-10',
                    help="Directory containing the dataset")
args = parser.parse_args()

model_dir = 'experiments/base_model'
train_log = os.path.join(model_dir, 'train_{}.log'.format(args.log))
test_log = os.path.join(model_dir, 'test_{}.log'.format(args.log))
for i in range(5, 10):
	train_script = 'CUDA_VISIBLE_DEVICES={} python main.py --train_range {} \
	 --loss_fn {} --finetune true --use_kfac {} --log _{} --data_dir {}'.format(args.gpu, i, args.loss_fn, \
	 	args.use_kfac, args.log, args.data_dir)
	os.system('echo {} >> {}'.format(train_script, train_log))
	os.system(train_script)
	test_fake_script = 'python evaluate.py --train_range {} \
	 --loss_fn {} --finetune true --use_kfac {} --log _{} --data_dir {}'.format(i, args.loss_fn, \
	 	args.use_kfac, args.log, args.data_dir)
	os.system('echo {} >> {}'.format(test_fake_script, test_log))
	test_script = 'CUDA_VISIBLE_DEVICES={} python evaluate.py --train_range [0-{}] \
	 --loss_fn {} --finetune true --use_kfac {} --log _{} --data_dir {}'.format(args.gpu, i, args.loss_fn, \
	 args.use_kfac, args.log, args.data_dir)	
	os.system(test_script)
	if args.use_kfac or args.loss_fn == 'retrain_regu_mine3':
		continue	
	collect_script = 'CUDA_VISIBLE_DEVICES={} python collect.py --train_range {} \
	 --loss_fn cnn --log _{} --data_dir {}'.format(args.gpu, i, args.log, args.data_dir)
	os.system(collect_script)
