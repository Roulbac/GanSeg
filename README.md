# GanSeg

GanSeg is a framework for medical image segmentation using various kinds of networks.

Currently GanSeg includes the following template generators: 

* UNet (Isola et al.)
* ResNet (Johnson et al.)
* CycleResNet (Zhu et al.)

Discriminators are all patch size discriminators with different patch sizes (1x1, 8x8, 16x16, 32x32, 64x64, etc...).

GanSeg also offers visualization classes and functions to display information such as histogram weights, dice coefficient, loss function value, sampled images, network computation graphs and so on.

GanSeg is able to slice medical volumes and their segmentations into axial, coronal and sagital views and process the slices in batches.

In test phase, GanSeg splits your volumes into slices, does its prediction and regroups the slices back in volumes with the same origin, spacing, dimensions and direction as your original volumes.


The functionality is easily extendable.

## Examples

![alt text](https://imgur.com/TZZk89p.png)

![alt text](https://imgur.com/cFsxx3u.png)

![alt text](https://imgur.com/RYbLpM4.png)



## Usage

Volumes in a dataset need to have consistent size accross the slicing dimensions, i.e. all slices need to have the same shape.

Place your datasets in the dataset folder and create subdirectories Images and Labels where you will store your images and their segmentations. The filenames need to be consistent accross Images and labels.

For training, here is an example commandline input:

`train.py --name test --vizport 8096 --model ganseg --dataset labvolslice --rootdir data/DS0 --image_shape 1,256,256 --gpu_ids 0 --gen_type unet_256 --loss nll --n_labels 2 train --lr 0.0001 --lr_policy lambda --batch_size 5 --n_epochs 5 --save_each 1 --decay_epoch 2 --log_freq 10`

Please follow train.py and test.py -h instructions for more detail.


## Requirements

* SimpleITK
* PyTorch
* NumPy
* Matplotlib
* Visdom

## Licenses


#### Third-party

[pytorchviz](https://github.com/szagoruyko/pytorchviz/blob/master/LICENSE)
