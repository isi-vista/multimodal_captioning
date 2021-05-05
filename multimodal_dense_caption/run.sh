#!/bin/bash

source ~/.bashrc

train=1
if [ $train -eq 1 ]; then
    data_dir=`pwd`/../augment_ms-coco-data/expts/
    model_dir=`pwd`/expts/models
    mkdir -p $model_dir

    python -u scps/train.py --feature_files $data_dir/feats/ms-coco-faster-rcnn/trn_cascade_fastrcnn_featex.pkl \
                                            $data_dir/feats/ms-coco-faster-rcnn/val_cascade_fastrcnn_featex.pkl \
                                            $data_dir/feats/multi30k-faster-rcnn/train_cascade_fastrcnn_featex.pkl \
                                            $data_dir/feats/multi30k-faster-rcnn/val_cascade_fastrcnn_featex.pkl \
                            --class_files $data_dir/classes/ms-coco-inceptionv3-classes/trn_inceptionv3_classes.pkl \
                                          $data_dir/classes/ms-coco-inceptionv3-classes/val_inceptionv3_classes.pkl \
                                          $data_dir/classes/multi30k-inceptionv3-classes/train_inceptionv3_classes.pkl \
                                          $data_dir/classes/multi30k-inceptionv3-classes/val_inceptionv3_classes.pkl \
                            --caption_files $data_dir/data/ms-coco-bpe/train.trans.pkl \
                                            $data_dir/data/ms-coco-bpe/val.trans.pkl \
                                            $data_dir/data/multi30k-bpe/train.trans.pkl \
                                            $data_dir/data/multi30k-bpe/val.trans.pkl \
                            --dense_caps_files ../dense_captioning/inputs/coco_train.bpe.trans.pkl \
                                               ../dense_captioning/inputs/coco_val.bpe.trans.pkl \
                                               ../dense_captioning/inputs/multi30k_train.bpe.trans.pkl \
                                               ../dense_captioning/inputs/multi30k_val.bpe.trans.pkl \
                            --tokenizer_file $data_dir/data/ms-coco-bpe/tokenizer.pkl \
                            --epochs 100 \
                            --batch_size 16 \
                            --saved_checkpoint $model_dir 


fi

