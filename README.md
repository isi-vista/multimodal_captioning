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
* run scripts in dense_captioning to train, evaluate the dense captioning model and use it to generate dense captions for MS-COCO dataset
* run training & evaluation experiment in multimodal_caption to train the model using detected objects' visual features, image classification features and source annotated captions (English)
* run training & evaluation experiment in multimodal_caption to train the model using detected objects' visual features, image classification features, dense captions associcated objects in the image, and source annotated captions (English)
