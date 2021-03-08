#!/bin/bash

source ~/.bashrc

# The users need to download and unzip the following datasets into the $data_root=/nas/multimudal/data/ms-coco or any other folder specified by $data_root:
#    http://images.cocodataset.org/zips/train2014.zip
#    http://images.cocodataset.org/zips/val2014.zip
#    http://images.cocodataset.org/annotations/annotations_trainval2014.zip

data_root=/nas/multimudal/data/ms-coco

# Load MS-COCO dataset
load_coco=1
if [ $load_coco -eq 1 ]; then
    annotation_file=$data_root/annotations/dataset_coco.json
    output_dir=`pwd`/expts/data/ms-coco

    mkdir -p $output_dir

    python -u scps/load_coco_data.py --input-annotation-file $annotation_file --input-image-dir $data_root --output-dir $output_dir 
fi


# Generating BPE captions for MS-COCO dataset
load_coco_bpe=0
if [ $load_coco_bpe -eq 1 ]; then
    annotation_file=$data_root/annotations/dataset_coco.json
    output_dir=`pwd`/expts/data/ms-coco-bpe
    mkdir -p $output_dir

    python -u scps/load_coco_data_with_bpe.py --input-annotation-file $annotation_file --input-image-dir $data_root --output-dir $output_dir
fi


# Run Faster-RCNN objects detection and feature extraction for MS-COCO dataset
faster_rcnn_feat=0
if [ $faster_rcnn_feat -eq 1 ]; then
    output_dir=`pwd`/expts/feats/faster-rcnn
    mkdir -p $output_dir

    export PYTHONPATH=scps/tensorpack/:${PYTHONPATH}

    for prefix in trn val test; do
        python -u scps/tensorpack/examples/FasterRCNN/featex.py --load scps/tensorpack/models/COCO-MaskRCNN-R101FPN9xGNCasAugScratch.npz \
                                                                --predict expts/data/ms-coco/${prefix}_img_key.pkl \
                                                                --output-filename ${prefix}_cascade_fastrcnn_featex1.pkl \
                                                                --output-dir $output_dir \
                                                                --config    TEST.RESULTS_PER_IM=64 TEST.RESULT_SCORE_THRESH_VIS=0 \
                                                                            FPN.CASCADE=True \
                                                                            MODE_MASK=False \
                                                                            BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3] \
                                                                            FPN.NORM=GN BACKBONE.NORM=GN \
                                                                            FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head \
                                                                            FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head \
                                                                            PREPROC.TRAIN_SHORT_EDGE_SIZE=[640,800] \
                                                                            TRAIN.LR_SCHEDULE=9x \
                                                                            BACKBONE.FREEZE_AT=0
    done 
fi


# Run inceptionv3 to extract features for detected objects 
inceptionv3_feat=0
if [ $inceptionv3_feat -eq 1 ]; then
    output_dir=`pwd`/expts/feats/inceptionv3
    mkdir -p $output_dir

    for prefix in trn val test; do
        python -u scps/inceptionv3_featex.py --predict expts/data/ms-coco/${prefix}_img_key.pkl \
                                             --output-filename ${prefix}_inceptionv3_featex.pkl \
                                             --output-dir $output_dir
    done 
fi


# Run Inception3 image classification to extract image class labels for MS-COCO dataset
inceptionv3_class=0
if [ $inceptionv3_class -eq 1 ]; then
    output_dir=`pwd`/expts/classes/inceptionv3
    mkdir -p $output_dir

    for prefix in val test trn; do
        python -u scps/inceptionv3_classification.py --predict expts/data/ms-coco/${prefix}_img_key.pkl \
                                                     --tokenizer expts/data/ms-coco/tokenizer.pkl \
                                                     --output-filename ${prefix}_inceptionv3_classes.pkl \
                                                     --output-dir $output_dir
    done
fi
