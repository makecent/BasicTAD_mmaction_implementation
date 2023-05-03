This is an un-official implementation of the [BasicTAD](https://github.com/MCG-NJU/BasicTAD) based on OpenMMlab repos (mmaction2).


# Prepare dataset
To use this repository, the user first need extract raw frames of THUMOS14. You may download the rawframes in this [link](https://connectpolyu-my.sharepoint.com/personal/19074484r_connect_polyu_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2F19074484r%5Fconnect%5Fpolyu%5Fhk%2FDocuments%2FDatasets%2FVideoDatasets%2Fthumos14%2Frawframes&ga=1)

# Prepare environment
```shell
conda create -n mmengine python=3.8 -y
conda activate mmengine
# the version of pytorch is flexible (>=2.0 is recommended), you may install pytorch depending on your machine.
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -U openmim fvcore future tensorboard pytorchvideo
mim install mmengine mmdet mmaction2
```

# Training
```shell
bash train.sh config/basicTAD_slowonly_96x10_1200e_thumos14_rgb.py 8
```