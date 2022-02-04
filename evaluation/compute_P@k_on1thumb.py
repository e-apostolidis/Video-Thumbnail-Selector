import sys
from choose_best_epoch import choose_epoch
import json
import numpy as np
import h5py

# structural similarity threshold
ssim_threshold = 0.7

exp_path = sys.argv[1]
dataset = sys.argv[2]
h5_file_path = '../data/' + dataset + '/' + dataset.lower() + '.h5'

best_epochs = choose_epoch(exp_path)
print('Selected models (training epochs)')
print(best_epochs)

hdf = h5py.File(h5_file_path, 'r')

P1_final = 0
P3_final = 0

for split in range(10):
	P1_split = 0
	P3_split = 0

	epoch = best_epochs[split]
	results_file = exp_path + '/results/split' + str(split) + '/' + dataset + '_' + str(epoch) + '.json'
	with open(results_file) as f:
		data = json.loads(f.read())
		video_names = list(data.keys())

	for video_name in video_names:
		# read the results (importance scores) for a video and sort them in descending order in order to find the frames with the max imp. scores
		imp_scores = np.asarray(data[video_name])
		sorted_score_inds = np.argsort(imp_scores)[::-1]

		# read the initial number of frames for the video (before sampling)
		n_frames = np.array(hdf.get(video_name+'/n_frames'))
		interval = round(n_frames/imp_scores.shape[0])
		top3_indices = []
		for index in sorted_score_inds[:3]:
			top3_indices.append(int(index*interval))

		# read the ssim matrix and the ground truth thumbnail-indices from the h5
		ssim_matrix = np.array(hdf.get(video_name+'/ssim_matrix'))
		top1_thumbnail_ids = np.array(hdf.get(video_name+'/top1_thumbnail_ids'))

		# compute P@1
		my_index = top3_indices[0]  # the top 1 thumbnail
		P1 = 0
		for gt_index in top1_thumbnail_ids:
			if ssim_matrix[my_index, gt_index] > ssim_threshold:
				P1 = 1
				break
		P1_split += P1

		# compute P@3
		P3 = 0
		for my_index in top3_indices[:3]:
			for gt_index in top1_thumbnail_ids:
				if ssim_matrix[my_index, gt_index] > ssim_threshold:
					P3 = 1
					break
			if P3 == 1:
				break
		P3_split += P3

	# find the P1 and P3 for each split
	P1_split = P1_split/len(video_names)
	P3_split = P3_split/len(video_names)

	P1_final += P1_split
	P3_final += P3_split

	print('Performance (P@1 & P@3) on split', split)
	print(P1_split, P3_split)

P1_final = P1_final/10
P3_final = P3_final/10

print('Average performance (P@1 & P@3) over all splits')
print(P1_final, P3_final)

hdf.close()
