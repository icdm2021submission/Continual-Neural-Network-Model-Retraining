import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', default='0',
                    help="gpu")
parser.add_argument('--data_dir', default='../data/cifar-10',
                    help="Directory containing the dataset")
parser.add_argument('--data', default='cifar-10',
                    help="Directory containing the dataset")

args = parser.parse_args()
use_bn = ''
data = args.data
data_dir = '../data/{}'.format(data)
# settings = ['dr']
settings = ['null', 'bn', 'dr', 'mr']
for setting in settings:
	log = 'test_{}'.format(setting)
	os.system('rm -r experiments')
	os.system('cp -r experiments_{}_{} experiments'.format(data, setting))
	if setting in ['bn', 'mr']:
		use_bn = '--use_bn true'
	os.system('CUDA_VISIBLE_DEVICES={} python evaluate.py --combine true --data_dir {} --log {} {}'.format(args.gpu, data_dir, log, use_bn))
	os.system('mv experiments/base_model/*.log result_{}_{}'.format(data, setting))


