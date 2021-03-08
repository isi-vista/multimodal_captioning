#!/bin/bash

source ~/.bashrc

# The users need to download multi30k dataset from https://github.com/multi30k/dataset to $multi30k_root=/nas/multimodal/data/multi30k/
# And download flickr30k images and annotations to $image_root=/nas/multimodal/data/Flickr30k

data_root=/nas/multimodal/data

# Load multi30k dataset without BPE
load_multi30k_task1=0
if [ $load_multi30k_task1 -eq 1 ]; then
    multi30k_root=$data_root/data/task1
    image_root=$data_root/Flickr30k/flickr30k_images
    image_list=$multi30k_root/image_splits/train_images.txt

    output_dir=`pwd`/expts/data/multi30k.task1

    mkdir -p $output_dir

    for prefix in train val test_2016_flickr; do
        python -u scps/load_multi30k_data.py --image_list $data_root/image_splits/${prefix}.txt \
                                             --tokenizer $output_dir/train.tokenizer.pkl \
                                             --en_trans $data_root/tok/${prefix}.lc.norm.tok.en \
                                             --de_trans $data_root/tok/${prefix}.lc.norm.tok.de \
                                             --image_dir $image_root --output_prefix $output_dir/${prefix}
    done

fi

# Load multi30k task1 dataset with BPE
load_multi30k_task1_bpe=0
if [ $load_multi30k_task1_bpe -eq 1 ]; then
    multi30k_root=$data_root/multi30k/data/task1
    image_root=$data_root/Flickr30k/flickr30k_images
    image_list=$multi30k_root/image_splits/train_images.txt
    output_dir=`pwd`/expts/data/multi30k.task1.bpe

    mkdir -p $output_dir

    for prefix in train val test_2016_flickr; do
        python -u scps/load_multi30k_data_with_bpe.py --image_list $multi30k_root/image_splits/${prefix}.txt \
                                             --tokenizer $output_dir/train.tokenizer.pkl \
                                             --codes $output_dir/train.codes.pkl \
                                             --en_trans $multi30k_root/tok/${prefix}.lc.norm.tok.en \
                                             --de_trans $multi30k_root/tok/${prefix}.lc.norm.tok.de \
                                             --image_dir $image_root --output_prefix $output_dir/${prefix}
    done

fi


# Run object detection and extract features for each object
faster_rcnn_feat=0
if [ $faster_rcnn_feat -eq 1 ]; then
    output_dir=`pwd`/expts/feats/faster-rcnn
    mkdir -p $output_dir

    export PYTHONPATH=scps/tensorpack/:${PYTHONPATH}

    for prefix in train val test_2016_flickr; do
        python -u scps/tensorpack/examples/FasterRCNN/featex.py --load scps/tensorpack/models/COCO-MaskRCNN-R101FPN9xGNCasAugScratch.npz \
                                                                --predict expts/data/multi30k.task1/${prefix}.image.list.pkl \
                                                                --output-filename ${prefix}_cascade_fastrcnn_featex.pkl \
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


# Image classification
inceptionv3_feat=0
if [ $inceptionv3_feat -eq 1 ]; then
    output_dir=`pwd`/expts/feats/inceptionv3
    mkdir -p $output_dir

    for prefix in train val test_2016_flickr; do
        python -u scps/inceptionv3_featex.py --predict expts/data/multi30k.task1/${prefix}.image.list.pkl \
                                             --output-filename ${prefix}_inceptionv3_featex.pkl \
                                             --output-dir $output_dir
    done 
fi


# Image classification
inceptionv3_class=0
if [ $inceptionv3_class -eq 1 ]; then
    output_dir=`pwd`/expts/classes/inceptionv3
    mkdir -p $output_dir

    for prefix in train val test_2016_flickr; do
        python -u scps/inceptionv3_classification.py --predict expts/data/multi30k.task1/${prefix}.image.list.pkl \
                                                     --tokenizer ../run.19/expts/data/ms-coco/tokenizer.pkl \
                                                     --output-filename ${prefix}_inceptionv3_classes.pkl \
                                                     --output-dir $output_dir
    done
fi
