import os

loss_fns = ['retrain_regu_mine2' , 'retrain_regu_fisher2']
# loss_fns = ['retrain_regu', 'retrain_regu_mas', 'retrain_regu_selfless', \
# 'retrain_regu_mine' , 'retrain_regu_fisher', 'cnn']

for loss_fn in loss_fns:
	script= 'python retrain_mini.py --loss_fn {} --log {}mini --gpu 0'.format(loss_fn, loss_fn.split('_')[-1])
	if loss_fn == 'cnn':
		script += ' --use_kfac true'
	os.system(script)
	os.system('mv experiments/base_model/*.log ./')
	os.system('rm -rf experiments')
	os.system('cp -r experiments_base experiments')
os.system('mv experiments/base_model/*.log ./')
