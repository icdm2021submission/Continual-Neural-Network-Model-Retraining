import glob
import pandas as pd
import os, sys

def getPaths(folder, prefix):
	txt_files = glob.glob(os.path.join(folder, '{}_*.log'.format(prefix)))
	return txt_files
	# for file_path in txt_files:
	# 	print(file_path)

# e.g., file_path = 'MQ2007_glrank.txt'
def get_data_model(file_path):
	fields = os.path.basename(file_path)
	dataset_model = fields.split('.')[0].split('_')
	return dataset_model[0], dataset_model[1]

def get_accs(file_path):
	with open(file_path, 'r') as f:
		accs = []
		lines = f.readlines()
		for line in lines:
			if '- Eval metrics:' in line:
			# if line.startswith('- Eval metrics:'):
				# print(line)
				fields = line.split(' ')
				acc = float(fields[-1])
				accs.append(acc)
		if (len(accs) != 5):
			print('ERROR! in ', file_path)
		return accs

def get_time(file_path):
	with open(file_path, 'r') as f:
		accs = []
		lines = f.readlines()
		for line in lines:
			if 'total time:' in line:
			# if line.startswith('total time:'):
				# print(line)
				fields = line.split(' ')
				acc = float(fields[-3])
				accs.append(acc)
		if (len(accs) != 5):
			print('ERROR! in {} with length: {}'.format(file_path, len(accs)))
		return accs

def print_tables(folder):
	print('-----------Results for {}'.format(folder))
	txt_files = getPaths(folder, 'test')
	txt_files.sort()
	lines = {}
	for file_path in txt_files:
		dataset, model = get_data_model(file_path)
		if 'kfac' in model:
			continue
		accs = get_accs(file_path)
		lines[model] = accs
		accs = ["{:.4f}".format(v) for v in accs]
		acc_str = ' &'.join(accs)
		# print('{} &{}'.format(model, acc_str))
	df = pd.DataFrame(lines)
	# print(df)
	run_model = df.idxmax(axis=1)
	# print(run_model)
	for model in lines:
		accs = lines[model]
		accs = ["{:.4f}".format(v) for v in accs]
		for i in range(len(accs)):
			if run_model[i] == model:
				accs[i] = '\\textbf{' + accs[i] + '}'
		acc_str = ' &'.join(accs)
		print('{} &{}\\\\ \\hline'.format(model, acc_str))

	# print()
	# txt_files = getPaths(folder, 'train')
	# txt_files.sort()
	# lines = {}
	# for file_path in txt_files:
	# 	dataset, model = get_data_model(file_path)
	# 	if 'kfac' in model:
	# 		continue		
	# 	accs = get_time(file_path)
	# 	lines[model] = accs
	# 	accs = ["{:.4f}".format(v) for v in accs]
	# 	acc_str = ' &'.join(accs)
	# 	# print('{} &{}'.format(model, acc_str))
	# df = pd.DataFrame(lines)
	# # print(df)
	# run_model = df.idxmin(axis=1)
	# # print(run_model)
	# for model in lines:
	# 	accs = lines[model]
	# 	accs = ["{:.4f}".format(v) for v in accs]
	# 	for i in range(len(accs)):
	# 		if run_model[i] == model:
	# 			accs[i] = '\\textbf{' + accs[i] + '}'
	# 	acc_str = ' &'.join(accs)
	# 	print('{} &{}\\\\ \\hline'.format(model, acc_str))

prefix = 'result_'
datasets = ['cifar-10']
if len(sys.argv) == 2:
	datasets = [sys.argv[1]]
folders = ['EI', 'EI2', 'EXP3', 'TS', 'UCB']

for dataset in datasets:
	print('\\subsection{' + dataset.upper() +' Results}')
	print()
	for folder in folders:
		# print('-------------------------')
		caption = '{} with the {} setting'.format(dataset, folder)
		folder = '{}{}_{}'.format(prefix, dataset, folder)
		# print(folder)
		# folder = '{}{}_{}'.format(prefix, folder, dataset)
		print_tables(folder)
		print()
		# print('-------------------------')
