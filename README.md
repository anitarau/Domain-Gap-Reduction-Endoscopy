# Domain-Gap-Reduction-Endoscopy



Anita Rau, Binod Bhattarai, Lourdes Agapito, Danail Stoyanov

[DEMI](https://demi-workshop.github.io) Workshop @ MICCAI 2023

![method_overview](teaser.png)
## Getting Started

```
cd Domain-Gap-Reduction-Endoscopy
mkdir outputs
mkdir saved_models
mkdir trained_models
```

## Pretrained Models

Download our pretrained model [here](https://drive.google.com/file/d/1DaUv-tZrijQimam1AF36gFvmTg6Vtxnz/view?usp=share_link) and save it to the ```trained_models``` folder.

## Real Data Setup

Apply for EndoMapper data access [here](https://www.synapse.org/#!Synapse:syn26707219/wiki/615178). Once you were granted access, navigate to Files --> Sequences --> Seq_001 --> meta-data --> colmap and add Seq_001.zip to your Download Cart. You can use python to download the data directly to your remote server:

```
pip install synapseclient
python
import synapseclient
syn = synapseclient.Synapse(cache_root_dir="/path/to/savedir")
syn.login("username","password")
dl_list_file_entities = syn.get_download_list()
```
If `cache_root_dir` is not specified, data is saved to `~/.synapseCache`.

The downloaded sequence is called '33' and has subfolders cluster_list and img_train. We undistorted all images in img_train and saved them to '33_undist/img_train'. Both '33' and '33_undist' should be in the `data_root` directory.


## Synthetic Data Setup

Download the synthetic dataset [here](https://rdr.ucl.ac.uk/articles/dataset/Simcol3D_-_3D_Reconstruction_during_Colonoscopy_Challenge_Dataset/24077763). To replicate this paper you only need to download SyntheticColon_I (not II and III).

Unzip the dataset in your `syn_data_root` directory. That's it!

## Testing

Once you have downloaded the real dataset and our pretrained model you can test on EndoMapper frames:

```
cd Domain-Gap-Reduction-Endoscopy
python src/test.py --data_root /path/to/data
```


## Training

Once you have downloaded the real and synthetic datasets you can train the model with:
```
cd Domain-Gap-Reduction-Endoscopy/src
python train.py --data_root /path/to/real/data --syn_data_root /path/to/synthetic/data --root_dir /path/to/outputs
```


## Cite our work if it helped you
```
@inproceedings{rau2023task,
  title={Task-Guided Domain Gap Reduction for Monocular Depth Prediction in Endoscopy},
  author={Rau, Anita and Bhattarai, Binod and Agapito, Lourdes and Stoyanov, Danail},
  booktitle={MICCAI Workshop on Data Engineering in Medical Imaging},
  pages={111--122},
  year={2023},
  organization={Springer}
}
```