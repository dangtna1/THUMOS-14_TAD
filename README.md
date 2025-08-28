# Dual-View Human Action Recognition (Egocentric + Exocentric Views)

## Datasets
The following table lists the benchmark datasets for the project:

| Dataset                                                    | Description                                                                                   |
| :--------------------------------------------------------- | :-------------------------------------------------------------------------------------------- |
| [THUMOS14](https://www.crcv.ucf.edu/THUMOS14/home.html)           | Consists of 413 videos with temporal annotations.                                           
| [Charades](https://prior.allenai.org/projects/charades)         | Contains dense-labeled 9,848 annotated videos of daily activities.                            |

## Up-to-date code
There are two branches:
* main: final code for late fusion
* exo-stream: final code for early fusion, one stream implementation
NOTE: For keep up-to-date to the project, Please refer to this [repo](https://github.com/dangtna1/THUMOS-14_TAD/tree/main)

## Sampling Charades dataset for experiments
Please refer to [this notebook](notebooks\charades_sample_building.ipynb)

## Pre-extracted features
* The publicly available one: 

|     Feature     |                                                                          Url                                                                           |         Backbone         |             Feature Extraction Setting             |
| :-------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------: | :------------------------------------------------: |
|      THUMOS i3d       |                           [Google Drive](https://drive.google.com/file/d/1iemRUtCVshYD3o9WahUrTTOW08GyCjfi/view?usp=sharing)                           |     I3D (two stream)     |    snippet_stride=4, extracted by ActionFormer     |
|      Charades i3d-rgb       |                           [Google Drive](https://drive.google.com/file/d/1iemRUtCVshYD3o9WahUrTTOW08GyCjfi/view?usp=sharing)                           |     I3D     |    24fps, snippet_stride=8, converted from [here](https://github.com/Xun-Yang/Causal_Video_Moment_Retrieval/blob/main/DATASET.md)     |

* Extracting features from I3D backbones yourself:
```bash
python .\tools\extract_features_full.py --ann_file data\charades\annotations\wise_annotations\combined_training_charades.json --save_dir data\complete_exo_ego_wise_v2
```

## Missing videos
In case there are missing videos or there are some issues extracting features for some videos that their features couldn't saved into .npy file. When training, this message could appear:
`FileNotFoundError: [Errno 2] No such file or directory: 'xxx/missing_files.txt`

To fix that, please run this to generate a `missing_files.txt`

```bash
python .\tools\generate_missing_list.py annotation.json feature_folder
```
## Config
Pay attention to the config file such as [this](configs\charades_i3d_rgb.py) to adjust these fields relevant to the target dataset:
* Number of classes (`num_classes` in `rpn_head`).
* Training/Testing splits (`subset_name`, `subset`).
* Experiments workspace (`sample_type`).
* And adjust `annotation_path`, `class_map`, `data_path` as well.
* Note: in main branch, instead of having `annotation_path`, we have distinguishable annotation files for egocentric `annotation_path_ego` and for exocentric `annotation_path_exo` for serving two-stream model with early/late fusion methodology.

## Training
Now that we have dataset, pre-extracted features from I3D, get ready to train our model with this example script:

```bash
python .\tools\train.py .\configs\charades_i3d_rgb.py                                     
```

We can continue training our model from a specific checkpoint (e.g. epoch_9) by this:
```bash
python .\tools\train.py .\configs\charades_i3d_rgb.py --resume exps\charades\actionformer_i3d_rgb\exo_only_wise\checkpoint\epoch_9.pth                  
```

## Testing
After training the model, having some certain checkpoints, the model is ready to inference videos in testing subset (make sure to have pre-extracted features for videos in this subset as well)

```bash
python .\tools\train.py .\configs\charades_i3d_rgb.py                                     
```

We can continue training our model from a specific checkpoint (e.g. epoch_9) by this:
```bash
python .\tools\test.py .\configs\charades_i3d_rgb.py --checkpoint exps\charades\actionformer_i3d_rgb\exo_only_wise\checkpoint\epoch_29.pth               
```

## Qualitative inspection
To have a broader look at specific videos' ground truths and model predictions, refer to [this notebook](notebooks\charades_analysis.ipynb) where I focus on plotting time-series chart to visualize the local difference between them of a specific video.