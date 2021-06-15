# EI2 62.05 0.63.33 0.64.17 0.64.28 0.64.68
# Subset 62.28 62.45 63.35 63.11 63.46
# Current 61.03 60.82 60.34 61.54 62.15


# EXP3 98.95 99.06 99.04 99.03 99.03
# Subset 99.03 99.02 99.07 99.17 99.04
# Current 98.91 98.96 99.02 99.11 99.01

settings = ['EI2', 'Subset', 'Current']

import numpy as np
import matplotlib.pyplot as plt

with open('cifar-10') as f:
	lines = f.readlines()

# data to plot
n_groups = 5
# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8
colors = ['b', 'g', 'r']
i = 0
for line in lines:
	fields = line.split()
	setting = fields[0]
	values = fields[1:]
	values = [float(v) for v in values]
	print(values)
	plt.bar(index + i * (bar_width), values, bar_width-0.05,
	alpha=opacity,
	color=colors[i],
	label=setting)
	i += 1

plt.xlabel('Settings')
plt.ylabel('Accuracy')
# plt.ylim([98.9,99.2])
plt.ylim([60,66])
plt.title('CIFAR-10- Best MAB compared to the subset and current settings')
plt.xticks(index + bar_width, ('R1', 'R2', 'R3', 'R4', 'R5'))
plt.legend()

plt.tight_layout()
plt.show()
