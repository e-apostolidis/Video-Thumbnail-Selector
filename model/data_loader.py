# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json

class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index):
        self.mode = mode
        self.name = video_type.lower()
        self.datasets = ['../data/OVP/ovp.h5',
                         '../data/Youtube/youtube.h5']
        self.splits_filename = ['../data/splits/' + self.name + '_splits.json']
        self.split_index = split_index # it represents the current split (varies from 0 to 9)

        if 'ovp' in self.splits_filename[0]:
            self.filename = self.datasets[0]
        elif 'youtube' in self.splits_filename[0]:
            self.filename = self.datasets[1]
        hdf = h5py.File(self.filename, 'r')
        self.list_features = []
        self.aesthetic_scores_mean = []

        with open(self.splits_filename[0]) as f:
            data = json.loads(f.read())
            for i, split in enumerate(data):
                if i == self.split_index:
                    self.split = split
                    
        for video_name in self.split[self.mode + '_keys']:
            features = torch.Tensor(np.array(hdf[video_name + '/features']))
            self.list_features.append(features)
            aes_scores_mean = torch.Tensor(np.array(hdf[video_name + '/aesthetic_scores_mean']))
            self.aesthetic_scores_mean.append(aes_scores_mean)

        hdf.close()

    def __len__(self):
        self.len = len(self.split[self.mode+'_keys'])
        return self.len

    # It returns the features, the video name and the frame-level aesthetics scores
    def __getitem__(self, index):
        video_name = self.split[self.mode + '_keys'][index]  #gets the current video name
        frame_features = self.list_features[index]
        aesthetic_scores_mean = self.aesthetic_scores_mean[index]
        return frame_features, video_name, aesthetic_scores_mean


def get_loader(mode, video_type, split_index):
    if mode.lower() == 'train':
        vd = VideoData(mode, video_type, split_index)
        return DataLoader(vd, batch_size=1, shuffle=True)
    else:
        return VideoData(mode, video_type, split_index)


if __name__ == '__main__':
    pass
