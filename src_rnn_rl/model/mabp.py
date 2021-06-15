"""Define the model."""
import sys, random, logging
import tensorflow as tf
import numpy as np
import random, math
from scipy import stats

def rl(params, sum_loss, numbers_of_selections, sums_of_reward, max_upper_bound, n, arm_weights):
	indexes = list(range(0, len(sum_loss)))
	np_sum_loss = np.array(sum_loss)
	if params.rl == 'Greedy':
	    chosen_arm = np.argmax(np_sum_loss)
	    reward = sum_loss[chosen_arm]
	    numbers_of_selections[chosen_arm] += 1
	    # logging.info('numbers_of_selections in mabp.py: {}'.format(numbers_of_selections))
	    return chosen_arm, reward, numbers_of_selections, None, None
	if params.rl == 'Reservoir':
	    chosen_arm = random.choice(indexes)
	    reward = sum_loss[chosen_arm]
	    numbers_of_selections[chosen_arm] += 1
	    # logging.info('numbers_of_selections in mabp.py: {}'.format(numbers_of_selections))
	    return chosen_arm, reward, numbers_of_selections, None, None	    
	if params.rl == 'EI':
	    if np.random.rand() < params.epilson:
	        chosen_arm = np.argmax(np_sum_loss)
	    else:
	        chosen_arm = random.choice(indexes)
	    reward = sum_loss[chosen_arm]
	    numbers_of_selections[chosen_arm] += 1
	    # logging.info('numbers_of_selections in mabp.py: {}'.format(numbers_of_selections))
	    return chosen_arm, reward, numbers_of_selections, None, None
	if params.rl == 'EI2':
		largest_index, largest2_index = top2(sum_loss)
		if np.random.rand() > params.beta:
		    if np.random.rand() < params.epilson:
		        chosen_arm = largest_index
		    else:
		        chosen_arm = random.choice(indexes)			
		else:
		    if np.random.rand() < params.epilson:
		        chosen_arm = largest2_index
		    else:
		        chosen_arm = random.choice(indexes)
		reward = sum_loss[chosen_arm]
		numbers_of_selections[chosen_arm] += 1
		return chosen_arm, reward, numbers_of_selections, None, None	    
	if params.rl == 'UCB':
		chosen_arm = np.argmax(np_sum_loss)
		for i in indexes:
			if (numbers_of_selections[i] > 0):
				average_reward = sums_of_reward[i] / numbers_of_selections[i]
				delta_i = math.sqrt(2 * math.log(n+1) / numbers_of_selections[i])
				upper_bound = average_reward + delta_i
			else:
				upper_bound = 1e400
			if upper_bound > max_upper_bound:
				max_upper_bound = upper_bound
		chosen_arm = i
		reward = sum_loss[chosen_arm]
		numbers_of_selections[chosen_arm] += 1
		sums_of_reward[chosen_arm] += sum_loss[chosen_arm]
		return chosen_arm, reward, numbers_of_selections, sums_of_reward, max_upper_bound
	if params.rl == 'TS':
		sum_ = sum(sum_loss)
		sum_loss = [float(v/sum_) for v in sum_loss]
		bandit_priors = [
		        stats.beta(a=1+w, b=1+t-w) for t, w in zip(numbers_of_selections, sums_of_reward)]
		# Sample a probability theta for each bandit
		theta_samples = [d.rvs(1) for d in bandit_priors]
		# logging.info(theta_samples)
		chosen_arm = np.argmax(theta_samples)
		logging.info(chosen_arm)
		reward = sum_loss[chosen_arm]
		numbers_of_selections[chosen_arm] += 1
		sums_of_reward[chosen_arm] += abs(reward)
		chosen_arm = 1
		return chosen_arm, reward, numbers_of_selections, sums_of_reward, None
	if params.rl == 'EXP3':
		n_arms = len(indexes)
		probs = [0.0 for i in indexes]
		total_weight = sum(arm_weights)
		for arm in indexes:
			probs[arm] = (1 - params.gamma) * (arm_weights[arm] / total_weight)
			probs[arm] = probs[arm] + params.gamma / float(n_arms)
		chosen_arm = np.argmax(probs)
		reward = sum_loss[chosen_arm]
		numbers_of_selections[chosen_arm] += 1
		x = reward / probs[chosen_arm]
		growth_factor = math.exp((params.gamma / n_arms) * x)
		arm_weights[chosen_arm] *= growth_factor
		return chosen_arm, reward, numbers_of_selections, arm_weights, None
	if params.rl == 'EXP4':
		n_arms = len(indexes)
		probs = [0.0 for i in indexes]
		total_weight = sum(arm_weights)
		for arm in indexes:
			probs[arm] = (1 - params.gamma) * (arm_weights[arm] / total_weight)
			probs[arm] = probs[arm] + params.gamma / float(n_arms)
		chosen_arm = np.argmax(probs)
		reward = sum_loss[chosen_arm]
		numbers_of_selections[chosen_arm] += 1
		x = reward / probs[chosen_arm]
		growth_factor = math.exp((params.gamma / n_arms) * x)
		arm_weights[chosen_arm] *= growth_factor
		return chosen_arm, reward, numbers_of_selections, arm_weights, None

def top2(list1):
    largest = list1[0]  
    largest_index = 0 
    largest2 = None
    largest2_index = -1
    for i in range(1, len(list1)):
    	item = list1[i]     
    	if item > largest:  
            largest2 = largest
            largest = item
            largest2_index = largest_index
            largest_index = i
    	elif largest2 == None or largest2 < item:  
            largest2 = item
            largest2_index = i
    	# print(largest, largest_index, largest2, largest2_index)
    return largest_index, largest2_index

def categorical_draw(probs):
	z = random.random()
	cum_prob = 0.0
	for i in range(len(probs)):
		prob = probs[i]
		cum_prob += prob
	if cum_prob > z:
		return i
	return len(probs) - 1
