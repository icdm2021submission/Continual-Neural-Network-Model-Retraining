import glob
import pandas as pd
import os, sys

log_model = {'retrain': 'Retrain','fishern': 'EWCN', 'fisher': 'EWC', 'mas': 'MAS', 'mine': 'New', \
'regu': 'Fine-tune', 'selfless': 'Selfless','mine4': 'NEW4', 'mine5': 'NEW5', 'mini4': 'MINI4', 'mini5': 'MINI5'}
def getPaths(folder, prefix):
	txt_files = glob.glob(os.path.join(folder, '{}_*.log'.format(prefix)))
	return txt_files

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
	# print('-----------Results for {}'.format(folder))
	txt_files = getPaths(folder, 'test')
	txt_files.sort()
	lines = {}
	for file_path in txt_files:
		dataset, model = get_data_model(file_path)
		if 'kfac' in model:
			continue
		# model = log_model[model]
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
	# 	model = log_model[model]
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
datasets = ['elec', 'sea']#, 'sea'
if len(sys.argv) == 2:
	datasets = [sys.argv[1]]
folders = ['0', '1', '2']
folder_type = {'0': 'union', '1': 'subset', '2':'current'}

for dataset in datasets:
	for folder in folders:
		print('-------------------------')
		print('{} with the {} setting'.format(dataset, folder_type[folder]))
		folder = '{}{}_{}'.format(prefix, dataset, folder)
		print(folder)
		print_tables(folder)
		print()
		print('-------------------------')

