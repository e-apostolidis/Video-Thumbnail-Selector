# Combining Adversarial and Reinforcement Learning for Video Thumbnail Selection

## PyTorch Implementation of the Video Thumbnail Selector
<div align="justify">

- From **"Combining Adversarial and Reinforcement Learning for Video Thumbnail Selection"** (accepted for publication at the IEEE ICMR 2021).
- Written by Evlampios Apostolidis, Eleni Adamantidou, Vasileios Mezaris and Ioannis Patras.
- This software can be used for training a deep learning architecture for video thumbnail selection after taking under consideration the representativeness and the aesthetic quality of the video frames. Training is performed in a fully-unsupervised manner based on a combination of adversarial and reinforcement learning. After being trained on a collection of videos, the Video Thumbnail Selector is capable of selecting a set of representative video thumbnails for unseen videos, according to a user-specified value about the number of required thumbnails. </div>

## Main dependencies
Tested, checked and verifiend with:
`Python` | `PyTorch` | `CUDA Version` | `TensorBoard` | `TensorFlow` | `NumPy` | `H5py`
:---:|:---:|:---:|:---:|:---:|:---:|:---:|
3.6 | 1.3.1 | 11.2 | 2.4.1 | 2.4.1 | 1.19.5 | 2.10.0

## Data
<div align="justify">

Structured h5 files with the video features and annotations of the OVP and Youtube datasets are available within the [data](https://github.com/e-apostolidis/Video_Thumbnail_Selector/tree/master/data) folder. These files have the following structure:
<pre>
/key
    /features                 2D-array with shape (n_steps, feature-dimension), feature vectors representing the content of the video frames; extracted from the pool5 layer of a GoogleNet trained on the ImageNet dataset
    /aesthetic_scores_mean    1D-array with shape (n_steps), scores representing the aesthetic quality of the video frames; computed as the softmax of the values in the final layer of a model of a [Fully Convolutional Network](https://github.com/bmezaris/fully_convolutional_networks) trained on the AVA dataset
    /n_frames                 number of video frames
    /ssim_matrix              2D-array with shape (top-5 selected thumbs, n_frames), the structural similarity scores between each of the five most selected thumbnails by the human annotators (in order to support evaluation using 'Precision at 5') and the entire frame sequence; computed using the [structural_similarity function](https://scikit-image.org/docs/dev/api/skimage.metrics.html#skimage.metrics.structural_similarity) of Python
    /top1_thumbnail_ids       the index of the most selected thumbnail by the human annotators (can be more than one if they exist more than one key-frames with the same ranking according to the number of selections made by the human annotators)
    /top3_thumbnail_ids       the indices of the three most selected thumbnails by the human annotators (can be more than three if they exist more than three key-frames with the same ranking according to the number of selections made by the human annotators)
</pre>
Original videos and human annotations (in the form of selected video thumbnails) for each dataset, are also available [here](https://sites.google.com/site/vsummsite/download). </div>

## Training
<div align="justify">

To train the model using one of the aforementioned datasets and for a number of randomly created splits of the dataset (where in each split 80% of the data is used for training and 20% for testing) use the corresponding JSON file that is included in the [data/splits](https://github.com/e-apostolidis/Video_Thumbnail_Selector/tree/master/data/splits) directory. This file contains the 10 randomly-generated splits that were utilized in our experiments.

For training the model using a single split, run:
```shell-script
python main.py --split_index N --n_epochs E --batch_size B --video_type 'dataset_name'
```
where, `N` refers to index of the used data split, `E` refers to the number of training epochs, `B` refers to the batch size, and `dataset_name` refers to the name of the used dataset.

Alternatively, to train the model for all 10 splits, use the [`run_ovp_splits.sh`](https://github.com/e-apostolidis/Video_Thumbnail_Selector/blob/master/model/run_ovp_splits.sh) and/or [`run_youtube_splits.sh`](https://github.com/e-apostolidis/Video_Thumbnail_Selector/blob/master/model/run_youtube_splits.sh) script and do the following:
```shell-script
chmod +x run_ovp_splits.sh    	# Makes the script executable.
chmod +x run_youtube_splits.sh  # Makes the script executable.
./run_ovp_splits                # Runs the script. 
./run_youtube_splits            # Runs the script.  
```
Please note that after each training epoch the algorithm performs an evaluation step, using the trained model to compute the estimated importance scores for the frames of each video of the test set. These scores are then used by the provided [evaluation](https://github.com/e-apostolidis/Video_Thumbnail_Selector/tree/master/evaluation) scripts to assess the overal performance of the model (in 'Precision at K').

The progress of the training can be monitored via the TensorBoard platform and by:
- opening a command line (cmd) and running: `tensorboard --logdir=/path/to/log-directory --host=localhost`
- opening a browser and pasting the returned URL from cmd. </div>

## Configurations
<div align="justify">

Setup for the training process:
 - In [`data_loader.py`](https://github.com/e-apostolidis/Video_Thumbnail_Selector/blob/master/model/data_loader.py), specify the path to the h5 file of the used dataset and the path to the JSON file containing data about the utilized data splits.
 - In [`configs.py`](https://github.com/e-apostolidis/Video_Thumbnail_Selector/blob/master/model/configs.py), define the directory where the analysis results will be saved to. </div>
   
Arguments in [`configs.py`](https://github.com/e-apostolidis/Video_Thumbnail_Selector/blob/master/model/configs.py): 
Parameter name | Description | Default Value | Options
| ---: | :--- | :---: | :---:
`--mode` | Mode for the configuration. | 'train' | 'train', 'test'
`--verbose` | Print or not training messages. | 'false' | 'true', 'false'
`--video_type` | Used dataset for training the model. | 'OVP' | 'OVP', 'Youtube'
`--input_size` | Size of the input feature vectors. | 1024 | -
`--hidden_size` | Size of the hidden representations. | 512 | int > 0
`--num_layers` | Number of layers of the LSTM units. | 2 | int > 0
`--n_epochs` | Number of training epochs. | 100 | int > 0
`--n_episodes` | Number of episodes for reinforcement learning. | 10 | int > 0
`--batch_size` | Size of the training batch, 40 for 'OVP' and 32 for 'Youtube'. | 40 | 0 < int ≤ len(Dataset)
`--clip` | Gradient norm clipping parameter. | 5 | float 
`--lr` | Learning rate for training all components besides the discriminator. | 1e-4 | float
`--discriminator_lr` | Learning rate for training the discriminator. | 1e-5 | float
`--split_index` | Index of the utilized data split. | 0 | 0 ≤ int ≤ 9
`--selected_thumbs` | Number of selected thumbnails. | 10 | int > 2

## Model Selection and Evaluation 
<div align="justify">

The utilized model selection criterion relies on the optimization of a core factor of the training process (i.e., the received reward) and enables the selection of a well-trained model by indicating the training epoch. To evaluate the trained models of the architecture and automatically select a well-trained model, run [`evaluate_exp.sh`](https://github.com/e-apostolidis/Video_Thumbnail_Selector/blob/master/evaluation/evaluate_exp.sh). To run this file, specify:
 - `$base_path`: the path to the folder where the analysis results are stored (e.g., '../data/results'),
 - `$exp_id`: the ID of the conducted experiment (e.g., 'exp1'), and
 - `$dataset_name`: the name of the utilized dataset ('OVP' or 'Youtube').

For further details about the adopted structure of directories in our implementation, please check line [#8 of evaluate_exp.sh](https://github.com/e-apostolidis/Video_Thumbnail_Selector/blob/master/evaluation/evaluate_exp.sh#L8). </div>

## Citation
If you find this implementation useful in your work, please cite the following publication where the corresponding method was proposed:

E. Apostolidis, E. Adamantidou, V. Mezaris and I. Patras, **"Combining Adversarial and Reinforcement Learning for Video Thumbnail Selection"**, in ACM Int. Conf. on Multimedia Retrieval (ICMR), Taipei, Taiwan, November 2021 (accepted for publication).

Bibtex:
<pre>
@InProceedings{Apostolidis_2021_ICMR,
    author    = {Apostolidis, Evlampios and Adamantidou, Eleni and Mezaris, Vasileios and Patras, Ioannis},
    title     = {Combining Adversarial and Reinforcement Learning for Video Thumbnail Selection},
    booktitle = {Proceedings of the ACM International Conference on Multimedia Retrieval (ICMR)},
    month     = {November},
    year      = {2021}
}
</pre>

## License
<div align="justify">
Copyright (c) 2021, Evlampios Apostolidis, Eleni Adamantidou, Vasileios Mezaris, Ioannis Patras / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
<div align="justify"> This work was supported by the EU Horizon 2020 programme under grant agreement H2020-832921 MIRROR, and by EPSRC under grant No. EP/R026424/1. </div>
