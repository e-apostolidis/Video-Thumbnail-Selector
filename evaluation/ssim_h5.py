from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from os import listdir
import h5py

# path where the frames of each video of the dataset are stored
frames_dir = '../<dataset_name>/video_frames'
# path where the thumbnails of each video of the dataset are stored
thumbnail_dir = '../<dataset_name>/thumbnail_images'

# h5 file where we will add info about the selected top-1/3/5 thumbnails and the ssim metric between them and each frame of the video
h5_file = '../<dataset_name>/<dataset_name>.h5'

# open h5 file to append information
hdf = h5py.File(h5_file, 'a')

for video_id in range(21,71): # for youtube
	print(video_id)
	top_thumb_dir = thumbnail_dir+'/v'+str(video_id)+'/top5'
	thumbnails = listdir(top_thumb_dir)
	thumbnails.sort()
	#print(thumbnails)
	frames = listdir(frames_dir+'/v'+str(video_id))
	frames.sort()
	#print(frames[:50])
	video_ssim = np.zeros((len(frames), len(thumbnails)), dtype=float)
	for (t_id, thumbnail) in enumerate(thumbnails):
		#print('thumbnail',thumbnail)
		thumb_image = cv2.imread(top_thumb_dir+'/'+thumbnail) # for each thumbnail, read the image
		gray_thumb_image = cv2.cvtColor(thumb_image, cv2.COLOR_BGR2GRAY) # convert to grayscale; the order of colors are BGR in opencv
		#print(gray_thumb_image.shape)
		for (f_id, frame) in enumerate(frames):
			#print('frame', frame)
			frame_image = cv2.imread(frames_dir+'/v'+str(video_id)+'/'+frame) # for each frame, read the image
			gray_frame_image = cv2.cvtColor(frame_image, cv2.COLOR_BGR2GRAY) # convert to grayscale
			#print(gray_frame_image.shape)
			video_ssim[f_id, t_id] = ssim(gray_thumb_image, gray_frame_image, full=False) # compute ssim and store it in the 'video_ssim' matrix

	# store 'video_ssim' in the h5 file
	hdf.create_dataset('v'+str(video_id)+'/ssim_matrix', data=video_ssim)

	top1_thumb_dir = thumbnail_dir+'/v'+str(video_id)+'/top1'
	thumbnails_top1 = listdir(top1_thumb_dir)
	top3_thumb_dir = thumbnail_dir+'/v'+str(video_id)+'/top3'
	thumbnails_top3 = listdir(top3_thumb_dir)

	thumbnails_top1_ids = []
	thumbnails_top3_ids = []
	for (i, t) in enumerate(thumbnails):
		if t in thumbnails_top1:
			thumbnails_top1_ids.append(i)
		if t in thumbnails_top3:
			thumbnails_top3_ids.append(i)

	# convert the lists with the thumbnail ids to numpy arrays and store them in the h5 file
	hdf.create_dataset('v'+str(video_id)+'/top1_thumbnail_ids', data=np.array(thumbnails_top1_ids))
	hdf.create_dataset('v'+str(video_id)+'/top3_thumbnail_ids', data=np.array(thumbnails_top3_ids))

	'''print(thumbnails_top1)
	print(thumbnails_top3)
	print(thumbnails_top1_ids)
	print(thumbnails_top3_ids)'''


hdf.close()

