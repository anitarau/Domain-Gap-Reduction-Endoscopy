# Domain-Gap-Reduction-Endoscopy



Anita Rau, Binod Bhattarai, Lourdes Agapito, Danail Stoyanov

[DEMI](https://demi-workshop.github.io) Workshop @ MICCAI 2023

## Pretrained Models

Download our pretrained model [here](https://drive.google.com/file/d/1DaUv-tZrijQimam1AF36gFvmTg6Vtxnz/view?usp=share_link) to ```saved_models``` folder.

## EndoMapper Data Setup

Apply for EndoMapper data access [here](https://www.synapse.org/#!Synapse:syn26707219/wiki/615178). Once you were granted access, navigate to Files --> Sequences --> Seq_001 --> meta-data --> colmap and add Seq_001.zip to your Download Cart. You can use python to download the data directly to your remote server:

```
pip install synapseclient
python
import synapseclient
syn = synapseclient.login('username','password')
dl_list_file_entities = syn.get_download_list()
```
By default data is saved to `~/.synapseCache`.

The downloaded sequence is called '33' and has subfolders cluster_list and img_train. We undistorted all images in img_train and saved them to '33_undist/img_train'. Both '33' and '33_undist' should be in the `data_root` directory.

## Testing

Test on EndoMapper frames:

```
cd Domain-Gap-Reduction-Endoscopy
python test.py --data_root /path/to/data
```


## Training

TO DO!