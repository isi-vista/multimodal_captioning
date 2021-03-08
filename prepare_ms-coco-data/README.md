Instructions to prepare MS-COCO dataset
============


In this module, we load the transcriptions of MS-COCO dataset and reorgnize it into the format that our experiments required, run the objects detection for each image and extract features for objects, and run image classification to extract image class labels.

Requirements
=============
* Python3, tensorflow
* cuda, cudnn
* Download ms-coco dataset, please refer run.sh
* Download pretrained model of tensorpack from http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R101FPN9xGNCasAugScratch.npz to scps/tensorpack/models/

