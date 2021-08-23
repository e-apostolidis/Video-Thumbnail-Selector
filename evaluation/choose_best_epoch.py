import csv
import torch

''' Chooses the best epoch based on a criterion (Received reward).
    Takes as input the path to .csv file with all the loss functions.
    Prints a scalar for each split that represents the selected epoch (starting from 0).'''


def choose_epoch(path):

	best_epochs = []

	for split in range(0, 10):
		logs_file = path+'/logs/split'+str(split)+'/scalars.csv'
		losses = {}
		losses_names = []

		with open(logs_file) as csv_file:
			csv_reader = csv.reader(csv_file, delimiter=',')

			for (i, row) in enumerate(csv_reader):
				if i == 0:
					for col in range(len(row)):
						losses[row[col]] = []
						losses_names.append(row[col])
				else:
					for col in range(len(row)):
						losses[losses_names[col]].append(float(row[col]))

		# criterion: Received reward
		reward = losses['reward_epoch']
		reward_t = torch.tensor(reward)

		# Normalize values
		reward_t = reward_t/max(reward_t)

		epoch = torch.argmax(reward_t)
		best_epochs.append(epoch.item())

	return best_epochs
