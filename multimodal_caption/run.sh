#!/bin/bash

source ~/.bashrc

train=1
if [ $train -eq 1 ]; then
    ms_coco_feature_dir=`pwd`/../prepare_ms-coco-data/expts
    multi30k_feature_dir=`pwd`/../prepare_multi30k-data/expts
    captions_dir=`pwd`/../augment_ms-coco-data/expts

    model_dir=`pwd`/expts/models
    mkdir -p $model_dir
    
    python -u scps/train.py --feature_files $ms_coco_feature_dir/feats/ms-coco-faster-rcnn/trn_cascade_fastrcnn_featex.pkl \
                                            $ms_coco_feature_dir/feats/ms-coco-faster-rcnn/val_cascade_fastrcnn_featex.pkl \
                                            $ms_coco_feature_dir/feats/ms-coco-faster-rcnn/test_cascade_fastrcnn_featex.pkl \
                                            $multi30k_feature_dir/feats/multi30k-faster-rcnn/train_cascade_fastrcnn_featex.pkl \
                                            $multi30k_feature_dir/feats/multi30k-faster-rcnn/val_cascade_fastrcnn_featex.pkl \
                                            $multi30k_feature_dir/feats/multi30k-faster-rcnn/test_2016_flickr_cascade_fastrcnn_featex.pkl \
                            --class_files $ms_coco_feature_dir/classes/ms-coco-inceptionv3-classes/trn_inceptionv3_classes.pkl \
                                          $ms_coco_feature_dir/classes/ms-coco-inceptionv3-classes/val_inceptionv3_classes.pkl \
                                          $ms_coco_feature_dir/classes/ms-coco-inceptionv3-classes/test_inceptionv3_classes.pkl \
                                          $multi30k_feature_dir/classes/multi30k-inceptionv3-classes/train_inceptionv3_classes.pkl \
                                          $multi30k_feature_dir/classes/multi30k-inceptionv3-classes/val_inceptionv3_classes.pkl \
                                          $multi30k_feature_dir/classes/multi30k-inceptionv3-classes/test_2016_flickr_inceptionv3_classes.pkl \
                            --caption_files $captions_dir/data/ms-coco-bpe/train.trans.pkl \
                                            $captions_dir/data/ms-coco-bpe/val.trans.pkl \
                                            $captions_dir/data/ms-coco-bpe/test.trans.pkl \
                                            $captions_dir/data/multi30k-bpe/train.trans.pkl \
                                            $captions_dir/data/multi30k-bpe/val.trans.pkl \
                                            $captions_dir/data/multi30k-bpe/test_2016_flickr.trans.pkl \
                            --tokenizer_file $captions_dir/data/ms-coco-bpe/tokenizer.pkl \
                            --epochs 100 \
                            --batch_size 16 \
                            --saved_checkpoint $model_dir 
fi

