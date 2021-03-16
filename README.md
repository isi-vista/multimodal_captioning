# multimodal_captioning

Requirements
=============
* Python3, tensorflow
* cuda, cudnn

Steps
=============
* run scripts in prepare_ms-coco-data to detect objects and extract visual features, class labels for images
* run scripts in prepare_multi30k-data to detect objects and extract visual features, class labels for images
* run scripts in augment_ms-coco-data to prepare transcriptions for ms-coco & multi30k dataset with BPE
* run training & evaluation experiment in multimodal_caption to train the model
