```shell
conda create -n mmengine python=3.8
# the version of pytorch is flexible, you may install pytorch depending on your machine.
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -U openmim fvcore future tensorboard
mim install mmengine mmdet mmaction2
```