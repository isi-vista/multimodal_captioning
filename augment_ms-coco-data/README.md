Instructions to augment MS-COCO dataset
============


In this module, we load the transcriptions of MS-COCO dataset and translate the English captions to German. Furthermore, we process the transcriptions of MS-COCO and Multi30k dataset with BPE.

Requirements
=============
* Python3, tensorflow
* cuda, cudnn
* Download and unpack pretrained model of fairseq from https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz to scps/fairseq/models/
